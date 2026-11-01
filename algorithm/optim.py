import math
import torch
from torch.optim import Optimizer

from torch import Tensor
from typing import List


def lmc(params: List[Tensor],
        d_p_list: List[Tensor],
        weight_decay: float,
        lr: float):
    r"""Functional API that performs Langevine MC algorithm computation.
    """

    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add_(param, alpha=weight_decay)

        param.add_(d_p, alpha=-lr)


class LangevinMC(Optimizer):
    def __init__(self,
                 params,              # parameters of the model
                 lr=0.01,             # learning rate
                 beta_inv=0.01,       # inverse temperature parameter
                 sigma=1.0,           # standard-deviation multiplier for Langevin noise
                 weight_decay=1.0,
                 device=None):   # l2 penalty
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive.")
        if not math.isfinite(beta_inv) or beta_inv < 0:
            raise ValueError("beta_inv must be finite and nonnegative.")
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and nonnegative.")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative.")
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.beta_inv = beta_inv
        self.lr = lr
        self.sigma = sigma
        self.temp = - math.sqrt(2 * beta_inv / lr) * sigma
        self.curr_step = 0
        defaults = dict(weight_decay=weight_decay)
        super(LangevinMC, self).__init__(params, defaults)

    def init_map(self):
        self.mapping = dict()
        index = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    num_param = p.numel()
                    self.mapping[p] = [index, num_param]
                    index += num_param
        self.total_size = index

    @torch.no_grad()
    def step(self):
        self.curr_step += 1
        for group in self.param_groups:
            parameters = []
            noisy_gradients = []
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                # Parameter-shaped noise follows the model's actual device and dtype.
                # The resulting parameter noise has scale sigma * sqrt(2 * lr * beta_inv).
                noise = self.temp * torch.randn_like(parameter)
                noisy_gradient = parameter.grad + noise
                parameters.append(parameter)
                noisy_gradients.append(noisy_gradient)
            lmc(parameters, noisy_gradients, group['weight_decay'], self.lr)
