import torch
from torch.optim import Optimizer

from .Visual import Visual


class Krylov(Optimizer, Visual):
    def __init__(self, params, lr=1e-4, beta=0.9, restart_interval=10):
        Optimizer.__init__(self, params, {'lr': lr, 'beta': beta, 'restart_interval': restart_interval})
        Visual.__init__(self)

        self.iteration = 0

    @torch.no_grad()
    def step(self, closure):
        loss = closure()
        self.loss_list.append(loss.item())

        self.iteration += 1

        for group in self.param_groups:
            lr = group['lr']
            k = group['restart_interval']

            for p in group['params']:
                if p.grad is None:
                    continue

                self.point_list.append(p.data.clone())

                grad = p.grad.data
                state = self.state[p]

                if 'prev_grad' not in state:
                    state['prev_grad'] = torch.zeros_like(p.data)
                    state['direction'] = -grad.clone()
                elif self.iteration % k == 0:
                    state['direction'] = -grad.clone()
                else:
                    prev_grad = state['prev_grad']
                    direction = state['direction']
                    beta_k = torch.dot(grad.flatten(), grad.flatten()) / (
                        torch.dot(prev_grad.flatten(), prev_grad.flatten()) + 1e-8)
                    direction = -grad + beta_k * direction
                    state['direction'] = direction

                p.data.add_(state['direction'], alpha=lr)
                state['prev_grad'] = grad.clone()

        return loss
