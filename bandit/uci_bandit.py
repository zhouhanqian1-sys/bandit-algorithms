import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


from sklearn.preprocessing import OrdinalEncoder
from sklearn.datasets import fetch_openml


from .base import Environment


continuous_dataset = ['shuttle', 'covertype']


def remove_nan(array: np.ndarray) -> np.ndarray:
    """Remove rows containing NaN values from *array* (2-D)."""
    mask = ~np.isnan(array).any(axis=1)
    return array[mask]




class UCI(Dataset):
    def __init__(self, datapath, dim_context, num_data=None, num_arms=2):
        super(UCI, self).__init__()
        self.dim_context = dim_context
        self.num_arms = num_arms
        self.loaddata(datapath, dim_context, num_data)

    def __getitem__(self, idx):
        x = self.context[idx]
        cxt = torch.zeros((self.num_arms, self.dim_context * self.num_arms))
        for i in range(self.num_arms):
            cxt[i, i * self.dim_context: (i + 1) * self.dim_context] = x
        return cxt, self.label[idx]

    def __len__(self):
        return self.label.shape[0]

    def loaddata(self, datapath, dim_context, num_data=None):
        data = np.loadtxt(datapath)
        self.label = (data[:, -1] - 1).astype(int)
        # data preprocessing
        context = data[:, 0:dim_context].astype(np.float32)
        if num_data:
            context = context[0:num_data]
            self.label = self.label[0:num_data]
        # context = context - context.mean(axis=0, keepdims=True)
        self.context = context / np.linalg.norm(context, axis=1, keepdims=True)
        self.context = torch.tensor(self.context)


class AutoUCI(Dataset):
    def __init__(self, name, dim_context, num_arms, num_data=None, version='active'):
        super(AutoUCI, self).__init__()
        self.dim_context = dim_context
        self.num_arms = num_arms
        self.loaddata(name, version, num_data)

    def __getitem__(self, idx):
        x = self.context[idx]
        cxt = torch.zeros((self.num_arms, self.dim_context * self.num_arms))
        for i in range(self.num_arms):
            cxt[i, i * self.dim_context: (i + 1) * self.dim_context] = x
        return cxt, self.label[idx]

    def __len__(self):
        return self.label.shape[0]

    def loaddata(self, name, version, num_data):
        cxt, label = fetch_openml(name=name, version=version, data_home='data', return_X_y=True)

        context = np.array(cxt).astype(np.float32)
        if num_data:
            label = label[0:num_data]
            context = context[0:num_data, :]
        # encode label
        # ensure numpy array to use reshape
        label = np.array(label)

        if name not in continuous_dataset:
            encoder = OrdinalEncoder(dtype=int)
            label = encoder.fit_transform(label.reshape(-1, 1))

            # Drop rows that contain Nan
            raw = np.concatenate([context, label], axis=1)
            raw = remove_nan(raw)
            self.label = raw[:, -1]
            context = raw[:, :-1]
        else:
            self.label = label.astype(int) - 1
        self.context = context / np.linalg.norm(context, axis=1, keepdims=True)
        self.context = torch.tensor(self.context)





class UCIBandit(Environment):
    """Contextual bandit based on an arbitrary OpenML (or local UCI) dataset.

    Each data point furnishes a context vector *x* ∈ ℝᵈ.  We construct *K*
    arms by placing *x* in a one-hot block along the feature dimension—exactly
    the same layout as :class:`AutoUCI` / :class:`UCI` datasets already
    implement.  Pulling arm *i* yields reward 1 iff *i* equals the ground-truth
    label, else 0.

    Parameters
    ----------
    name : str | PathLike
        Dataset name recognised by OpenML. If you prefer to load from a local
        text file, pass the *filepath* here **and** set ``use_auto=False``.
    dim_context : int
        Dimensionality of raw feature vector for each sample.
    num_arms : int
        Number of categorical classes ⇒ number of arms per round.
    num_data : int | None, default None
        Limit the total number of samples to this many. ``None`` keeps all.
    version : str | int, default "active"
        OpenML dataset version.
    device : str, default "cpu"
        Device on which to place context tensors.
    use_auto : bool, default True
        If ``True`` we fetch data via :class:`AutoUCI`; else we read a local
        whitespace-separated text file via :class:`UCI`.
    batch_size : int, default 1
        Internally we iterate through the dataset via a :class:`~torch.utils.data.DataLoader`.
        Keep this at 1 to mimic sequential bandit rounds.
    """

    def __init__(
        self,
        name: str,
        dim_context: int,
        num_arms: int,
        num_data: int | None = None,
        version: str | int = "active",
        device: str = "cpu",
        use_auto: bool = True,
        batch_size: int = 1,
    ) -> None:

        super().__init__()

        self.K = num_arms
        self.d = dim_context * num_arms  # feature dimension after concatenation
        self.device = device

        if use_auto:
            self.dataset = AutoUCI(name, dim_context, num_arms, num_data, version)
        else:
            self.dataset = UCI(name, dim_context, num_data, num_arms)

        # default to full dataset length when num_data is None
        self.n_contexts = len(self.dataset)

        # Loader yields one sample per round in random order to simulate online rounds
        self.loader = DataLoader(self.dataset, shuffle=True, batch_size=batch_size)
        self._iter = iter(self.loader)

        # Placeholders updated by get_action_set()
        self.current_label: int | None = None
        self.current_action_set: torch.Tensor | None = None  # shape (K, d)

        # Prime first context
        self.get_action_set()

    # ------------------------------------------------------------------ API
    def reset(self):
        """Start a fresh shuffled data pass before an independent experiment run."""
        self._iter = iter(self.loader)
        self.current_label = None
        self.current_action_set = None

    def get_action_set(self) -> torch.Tensor:
        """Return context matrix of shape ``(K, d)`` for current round."""

        try:
            cxt, label = next(self._iter)
        except StopIteration:
            # Restart iteration once dataset exhausted
            self._iter = iter(self.loader)
            cxt, label = next(self._iter)

        # Remove DataLoader batch dimension and move to device
        cxt = cxt.squeeze(0).to(self.device)  # (K, d)
        self.current_action_set = cxt
        self.current_label = int(label.item())
        action_set = cxt.detach().cpu().numpy()
        return action_set

    def get_reward(self, action_index: int) -> int:
        """Binary reward: 1 iff chosen arm matches true label."""
        assert self.current_label is not None, "Call get_action_set() before rewarding."
        return 1 if action_index == self.current_label else 0

    def get_means(self) -> np.ndarray:
        """Expected reward vector (one-hot on true label)."""
        means = np.zeros(self.K, dtype=float)
        if self.current_label is not None:
            means[self.current_label] = 1.0
        return means

    # ------------------------------------------------------------------ misc
    def name(self) -> str:  # noqa: D401 (simple name)
        return "UCIBandit"