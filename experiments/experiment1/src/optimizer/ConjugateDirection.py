from typing import Literal

import torch
from torch.optim import Optimizer

from .Visual import Visual


class ConjugateDirection(Optimizer, Visual):
    def __init__(self, params, lr=1e-2, beta_type: Literal['FR', 'PR'] = 'FR'):
        Optimizer.__init__(self, params, {'lr': lr, 'beta_type': beta_type})
        Visual.__init__(self)

        self.prev_grads = None
        self.prev_dirs = None

    @torch.no_grad()
    def step(self, closure):
        loss = closure()
        self.loss_list.append(loss.item())

        grads = []
        params_with_grad = []
        for group in self.param_groups:
            beta_type = group['beta_type']

            for p in group['params']:
                if p.grad is None:
                    continue

                self.point_list.append(p.data.clone())

                grad = p.grad.detach().clone()
                grads.append(grad)
                params_with_grad.append(p)

        flat_grad = torch.cat([g.view(-1) for g in grads])
        grad_norm_sq = flat_grad @ flat_grad

        if self.prev_grads is None:
            direction = -flat_grad
        else:
            prev_grad = self.prev_grads
            prev_dir = self.prev_dirs
            if beta_type == 'FR':
                beta = grad_norm_sq / (prev_grad @ prev_grad + 1e-10)
            elif beta_type == 'PR':
                y = flat_grad - prev_grad
                beta = (flat_grad @ y) / (prev_grad @ prev_grad + 1e-10)
            direction = -flat_grad + beta * prev_dir

        offset = 0
        for p, g in zip(params_with_grad, grads):
            numel = p.numel()
            d = direction[offset:offset + numel].view_as(p)
            p.add_(group['lr'] * d)
            offset += numel

        self.prev_grads = flat_grad.clone()
        self.prev_dirs = direction.clone()
        self.zero_grad()

        return loss
