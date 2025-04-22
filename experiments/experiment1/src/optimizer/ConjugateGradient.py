import torch
from torch.optim import Optimizer

from .Visual import Visual


class ConjugateGradient(Optimizer, Visual):
    def __init__(self, params, lr=1e-3):
        Optimizer.__init__(self, params, {'lr': lr})
        Visual.__init__(self)

        self.state['prev_grads'] = None
        self.state['prev_dirs'] = None

    @torch.no_grad()
    def step(self, closure):
        loss = closure()
        self.loss_list.append(loss.item())

        prev_grads = self.state.get('prev_grads')

        new_grads = []
        new_dirs = []

        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                self.point_list.append(p.data.clone())

                grad = p.grad.detach()

                if prev_grads is None:
                    direction = -grad
                else:
                    prev_grad = self.state['prev_grads'].pop(0)
                    prev_dir = self.state['prev_dirs'].pop(0)
                    beta = grad.dot(grad - prev_grad) / (prev_grad.dot(prev_grad) + 1e-10)
                    direction = -grad + beta * prev_dir

                p.add_(lr * direction)

                new_grads.append(grad)
                new_dirs.append(direction)

        self.state['prev_grads'] = new_grads
        self.state['prev_dirs'] = new_dirs

        return loss
