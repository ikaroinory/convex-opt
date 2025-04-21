import torch
from torch.optim import Optimizer

from .Visual import Visual


class SubgradientDescent(Optimizer, Visual):
    def __init__(self, params, lr=1e-3):
        Optimizer.__init__(self, params, {'lr': lr})
        Visual.__init__(self)

    @torch.no_grad()
    def step(self, closure):
        loss = closure()
        self.loss_list.append(torch.round(loss, decimals=4).item())

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue

                self.point_list.append(p.data.clone())

                d_p = p.grad.data
                p.data -= lr * d_p

        return loss
