
import numpy as np
from collections import defaultdict
from copy import deepcopy
from scipy.stats import bernoulli
from math import log
import seaborn as sns
colors = sns.color_palette('colorblind')
import random
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset






def _environment_call(function, seed, *args):
    """Keep context and reward randomness separate from agent exploration."""
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        np.random.seed(seed)
        # The supplied environments use NumPy and the CPU DataLoader generator.
        torch.random.default_generator.manual_seed(seed)
        return function(*args)
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def play_bandit(environment, agent, N, T, seed=42):
    """Collect instantaneous pseudo-regret for N independent runs of T rounds."""
    if N < 1 or T < 1:
        raise ValueError("N and T must be positive.")
    data = np.zeros((N, T))
    run_sequences = np.random.SeedSequence(seed).spawn(N)

    for run_index, run_sequence in enumerate(run_sequences):
        # Each run gets fresh randomness; algorithms receive matching environment seeds.
        run_seeds = run_sequence.generate_state(2 * T + 2)
        agent_seed = int(run_seeds[0])
        random.seed(agent_seed)
        np.random.seed(agent_seed)
        torch.manual_seed(agent_seed)
        agent.reset()
        if hasattr(environment, "reset"):
            _environment_call(environment.reset, int(run_seeds[1]))

        for round_index in range(T):
            context_seed = int(run_seeds[2 * round_index + 2])
            reward_seed = int(run_seeds[2 * round_index + 3])
            action_set = _environment_call(environment.get_action_set, context_seed)
            action_index, action = agent.get_action(action_set)
            if environment.name() == "UCIBandit":
                reward = _environment_call(environment.get_reward, reward_seed, action_index)
            else:
                reward = _environment_call(environment.get_reward, reward_seed, action)
            context = action_set[action_index]
            agent.receive_reward(action, context, reward)

            # Regret uses expected rewards, not noisy observed rewards.
            means = environment.get_means()
            data[run_index, round_index] = np.max(means) - means[action_index]

    return agent.name(), data


def run_experiment(environment, agents, N, T, seed=42):
    """Compare agents using independent runs and shared environment randomness."""
    names = [agent.name() for agent in agents]
    if len(names) != len(set(names)):
        raise ValueError("Agent names must be unique so results cannot overwrite each other.")
    if not agents:
        raise ValueError("Provide at least one agent to compare.")

    # Experiment execution should not change the caller's random streams.
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    all_data = {}
    try:
        for agent in agents:
            agent_id, regrets = play_bandit(environment, agent, N, T, seed=seed)
            all_data[agent_id] = regrets
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    return all_data

def plot_result(regrets, logscale=False, lb=None,q=10, mode = 'regret', cumulative=True):
    """
    regrets must be a dict {'agent_id':regret_table}
    """
    
    reg_plot = plt.figure()
    #compute useful stats
#     regret_stats = {}
    for i, agent_id in enumerate(regrets.keys()):
        data = regrets[agent_id]
        N, T = data.shape
        if cumulative:
            cumdata = np.cumsum(data, axis=1) # cumulative regret
        else:
            cumdata = data
        
        mean_reg = np.mean(cumdata, axis=0)
        q_reg = np.percentile(cumdata, q, axis=0)
        Q_reg = np.percentile(cumdata, 100-q, axis=0)
        
#         regret_stats[agent_id] = np.array(mean_reg, q_reg, Q_reg)
        
        plt.plot(np.arange(T), mean_reg, color=colors[i], label=agent_id)
        plt.fill_between(np.arange(T), q_reg, Q_reg, color=colors[i], alpha=0.2)
        
    if logscale:
        plt.xscale('log')
        plt.xlim(left=100)

    if lb is not None: 
        plt.plot(np.arange(T), lb, color='black', marker='*', markevery=int(T/10))
        
    plt.xlabel('time steps')
    header = "Cumulative " if cumulative else ""
    plt.ylabel(header + mode.capitalize())
    plt.legend()
    #reg_plot.show()
    return reg_plot #notebook version





class Collector(Dataset):
    '''
    Collect the context vectors that have appeared 
    '''

    def __init__(self):
        super(Collector, self).__init__()
        self.context = []
        self.rewards = []
        self.chosen_arms = []

    def __getitem__(self, key):
        return self.context[key], self.rewards[key]

    def __len__(self):
        return len(self.rewards)

    def collect_data(self, arm, context, reward):
        # Runners supply NumPy features; tensor callers are also accepted.
        context_tensor = torch.as_tensor(context, dtype=torch.float32).detach().cpu().clone()
        self.context.append(context_tensor)
        self.chosen_arms.append(arm)
        self.rewards.append(float(reward))

    def fetch_batch(self, batch_size=None):
        if batch_size is None:
            return self.context, self.rewards
        if batch_size < 1:
            raise ValueError("batch_size must be positive or None.")
        if batch_size >= len(self.rewards):
            return self.context, self.rewards
        # Include the last legal starting point and allow batch_size == history size.
        offset = np.random.randint(0, len(self.rewards) - batch_size + 1)
        return self.context[offset:offset + batch_size], self.rewards[offset:offset + batch_size]

    def clear(self):
        self.context = []
        self.rewards = []
        self.chosen_arms = []




def sample_data(dataloader):
    while True:
        for batch in dataloader:
            yield batch




def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("state value:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print("policy:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("State value:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # Print with width 6 for better alignment
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print("Policy:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # Goal state
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()