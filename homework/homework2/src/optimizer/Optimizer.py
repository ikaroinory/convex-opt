from typing import Callable

import torch


class Optimizer:
    def __init__(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        grad_f: Callable[[torch.Tensor], torch.Tensor] = None,
        epsilon=1e-6,
        max_iter=100
    ):
        self.f = f
        self.grad_f = grad_f
        self.epsilon = epsilon
        self.max_iter = max_iter

        self.iterator_count = 0

    def __call__(self, *args, **kwargs):
        pass
