import torch
from torch.optim import Optimizer

from .Visual import Visual


class StochasticGradientDescent(Optimizer, Visual):
    def __init__(self, params, lr=1e-3, momentum=0.0):
        Optimizer.__init__(self, params, {'lr': lr, 'momentum': momentum})
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
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                self.point_list.append(p.data)

                d_p = p.grad

                # 获取或创建动量缓存
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p)  # v = μ*v + grad

                p.data.add_(buf, alpha=-lr)

        return loss
