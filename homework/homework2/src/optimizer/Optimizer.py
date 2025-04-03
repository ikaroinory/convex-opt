from typing import Callable

import torch


class Optimizer:
    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor], grad_f: Callable[[torch.Tensor], torch.Tensor]):
        self.f = f
        self.grad_f = grad_f

        self.iterator_count = 0

    def __call__(self, *args, **kwargs):
        pass
