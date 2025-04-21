import torch
from torch.optim import Optimizer

from .Visual import Visual


class BFGS(Optimizer, Visual):
    def __init__(self, params, lr=1.0):
        Optimizer.__init__(self, params, {'lr': lr})
        Visual.__init__(self)

        self.state['old_params'] = None
        self.state['old_grads'] = None
        self.state['H'] = None

    @torch.no_grad()
    def step(self, closure):
        loss = closure()
        self.loss_list.append(loss.item())

        params_flat, grads_flat = self._gather_flat_params_and_grads()

        state = self.state
        lr = self.param_groups[0]['lr']

        if state['H'] is None:
            state['H'] = torch.eye(len(grads_flat), dtype=grads_flat.dtype, device=grads_flat.device)

        if state['old_params'] is not None:
            s = params_flat - state['old_params']
            y = grads_flat - state['old_grads']

            rho = 1.0 / (y @ s + 1e-10)
            I = torch.eye(len(s), device=s.device)
            H = state['H']

            H = (I - rho * s[:, None] @ y[None, :]) @ H @ (I - rho * y[:, None] @ s[None, :]) + rho * s[:, None] @ s[None, :]
            state['H'] = H

        direction = -state['H'] @ grads_flat
        new_params = params_flat + lr * direction
        self._set_flat_params(new_params)

        state['old_params'] = params_flat
        state['old_grads'] = grads_flat

        return loss

    def _gather_flat_params_and_grads(self):
        flat_params = []
        flat_grads = []
        for group in self.param_groups:
            for p in group['params']:
                self.point_list.append(p.data.clone())

                flat_params.append(p.data.view(-1))
                flat_grads.append(p.grad.view(-1))
        return torch.cat(flat_params), torch.cat(flat_grads)

    def _set_flat_params(self, flat_tensor):
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.numel()
                p.data.copy_(flat_tensor[idx:idx + numel].view_as(p))
                idx += numel
