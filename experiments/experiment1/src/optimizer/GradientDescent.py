import torch
from torch.optim.optimizer import Optimizer

from .Visual import Visual


class GradientDescent(Optimizer, Visual):
    def __init__(self, params, lr=1e-3):
        Optimizer.__init__(self, params, {'lr': lr})
        Visual.__init__(self)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            self.loss_list.append(torch.round(loss, decimals=4).item())

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue

                self.point_list.append(p.data.clone())

                p.data -= lr * p.grad

        return loss
