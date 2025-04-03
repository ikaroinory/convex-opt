from typing import Callable

import torch


class Optimizer:
    def __init__(
        self,
        f: Callable[[torch.Tensor], torch.Tensor],
        grad_f: Callable[[torch.Tensor], torch.Tensor] = None,
        epsilon=None,
        max_iter=None
    ):
        self.f = f
        self.grad_f = grad_f
        self.epsilon = epsilon if epsilon is not None else 1e-6
        self.max_iter = max_iter if max_iter is not None else 100

        self.iterator_count = 0
