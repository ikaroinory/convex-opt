import torch
from torch.optim import Optimizer

from .Visual import Visual


class DampedNewton(Optimizer, Visual):
    def __init__(self, params, lr=1.0, damping=1e-2):
        Optimizer.__init__(self, params, {'lr': lr, 'damping': damping})
        Visual.__init__(self)

    def step(self, closure):
        loss = closure()
        self.loss_list.append(loss.item())

        for group in self.param_groups:
            lr = group['lr']
            damping = group['damping']

            for p in group['params']:
                if p.grad is None:
                    continue

                self.point_list.append(p.data.clone())

                grad = p.grad.view(-1)

                def hvp_fn(v):
                    hv = torch.autograd.grad(
                        outputs=torch.autograd.grad(loss, p, create_graph=True)[0],
                        inputs=p,
                        grad_outputs=v,
                        retain_graph=True
                    )[0]
                    return hv.view(-1)

                v = -grad
                hv = hvp_fn(v)
                hv_damped = hv + damping * v

                update = lr * v / (hv_damped.norm() + 1e-8)
                p.data.add_(update.view_as(p))

        self.zero_grad()
        return loss
