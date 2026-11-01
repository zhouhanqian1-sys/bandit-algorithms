import torch
import torch.nn as nn



class MLP(nn.Module):
    def __init__(self, num_arms, dim_context, layers=None, act='LeakyReLU', norm=False):
        super(MLP, self).__init__()

        self.norm = nn.LayerNorm(
            dim_context, elementwise_affine=False) if norm else None

        self.act_dict = nn.ModuleDict(
            {
                'LeakyReLU': nn.LeakyReLU(),
                'ReLU': nn.ReLU(),
                'tanh': nn.Tanh(),
                'elu': nn.ELU(),
                'selu': nn.SELU(),
                'glu': nn.GELU()
            }
        )

        if layers is None:
            layers = [dim_context, 50]
        else:
            layers = [dim_context] + layers
        network = []
        for in_size, out_size in zip(layers, layers[1:]):
            network += [
                nn.Linear(in_size, out_size, bias=True),
                self.act_dict[act]
            ]
        network += [nn.Linear(layers[-1], num_arms, bias=False)]
        self.net = nn.Sequential(*network)
    
    def forward(self, x):
        if self.norm:
            x = self.norm(x)
        output = self.net(x)
        return output

    def init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                module.reset_parameters()