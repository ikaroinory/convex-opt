import torch
from torch.optim import Optimizer

from .Visual import Visual


class ADMM(Optimizer, Visual):
    def __init__(self, params, lr=1e-3, rho=1e-2, l1_weight=1e-4):
        Optimizer.__init__(self, params, {'lr': lr, 'rho': rho, 'l1_weight': l1_weight})
        Visual.__init__(self)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['z'] = torch.zeros_like(p.data)
                state['u'] = torch.zeros_like(p.data)

    def step(self, closure):
        loss = closure()
        self.loss_list.append(loss.item())

        for group in self.param_groups:
            lr = group['lr']
            rho = group['rho']
            l1_weight = group['l1_weight']

            for p in group['params']:
                if p.grad is None:
                    continue

                self.point_list.append(p.data.clone())

                grad = p.grad.data
                state = self.state[p]
                z = state['z']
                u = state['u']

                p.data = p.data - lr * (grad + rho * (p.data - z + u))

                x_hat = p.data + u
                state['z'] = self._soft_thresholding(x_hat, l1_weight / rho)

                state['u'] = u + p.data - state['z']

        return loss

    def _soft_thresholding(self, x, lamb):
        return torch.sign(x) * torch.clamp(torch.abs(x) - lamb, min=0.0)
