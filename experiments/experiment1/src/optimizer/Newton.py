import torch
from torch.optim.optimizer import Optimizer

from .Visual import Visual


class Newton(Optimizer, Visual):
    def __init__(self, params):
        Optimizer.__init__(self, params, {})
        Visual.__init__(self)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            self.loss_list.append(loss.item())

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                self.point_list.append(p.data.clone())

                grad = p.grad.data

                flat_param = p.view(-1)

                grad_output = torch.autograd.grad(loss, p, create_graph=True)[0]
                hessian = []

                for i in range(flat_param.numel()):
                    grad2 = torch.autograd.grad(grad_output.view(-1)[i], p, retain_graph=True)[0]
                    hessian.append(grad2.view(-1))

                H = torch.stack(hessian)

                H += 1e-4 * torch.eye(H.size(0))
                H_inv = torch.inverse(H)

                update = H_inv @ grad.view(-1)
                p.data -= update.view(p.size())

        return loss
