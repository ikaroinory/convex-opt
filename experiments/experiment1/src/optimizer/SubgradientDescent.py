import torch
from torch.optim import Optimizer

from optimizer.Visual import Visual


class SubgradientDescent(Optimizer, Visual):
    def __init__(self, params, lr=1e-3):
        Optimizer.__init__(self, params, {'lr': lr})
        Visual.__init__(self)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                self.loss_list.append(loss.item())

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue

                self.point_list.append(p.data)

                d_p = p.grad.data
                p.data -= lr * d_p

        return loss
