import torch
from torch.optim import Optimizer

from .Visual import Visual


class RandomSearch(Optimizer, Visual):
    def __init__(self, params, lr=1e-2):
        Optimizer.__init__(self, params, {'lr': lr})
        Visual.__init__(self)

        self.best_params = [p.clone().detach() for group in self.param_groups for p in group['params']]
        self.best_loss = float('inf')

    @torch.no_grad()
    def step(self, closure):
        loss = closure()

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = [p.clone().detach() for group in self.param_groups for p in group['params']]

        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.requires_grad:
                    noise = torch.randn_like(p) * lr
                    p.add_(noise)

        new_loss = closure()

        if new_loss >= self.best_loss:
            idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    p.data.copy_(self.best_params[idx])

                    self.loss_list.append(loss.item())
                    self.point_list.append(p.data.clone())

                    idx += 1
        else:
            self.best_loss = new_loss
            self.best_params = [p.clone().detach() for group in self.param_groups for p in group['params']]

        return self.best_loss
