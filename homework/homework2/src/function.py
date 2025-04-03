from typing import Callable

import torch


def f1(data: torch.Tensor) -> torch.Tensor:
    x = data[..., 0]
    y = data[..., 1]

    return x - y + 2 * x ** 2 + 2 * x * y + y ** 2


def f1_grad(data: torch.Tensor) -> torch.Tensor:
    x = data[..., 0]
    y = data[..., 1]

    return torch.stack([1 + 4 * x + 2 * y, -1 + 2 * x + 2 * y], dim=-1)


def f2a(x: torch.Tensor) -> torch.Tensor:
    return 2 * x ** 2 - x - 1


f2a_grad: Callable[[torch.Tensor], torch.Tensor] = lambda x: 4 * x - 1

f2b: Callable[[torch.Tensor], torch.Tensor] = lambda x: 3 * x[..., 0] ** 2 - 21.6 * x[..., 0] - 1
f2b_grad: Callable[[torch.Tensor], torch.Tensor] = lambda x: 6 * x[..., 0] - 21.6


def f3(data: torch.Tensor) -> torch.Tensor:
    x = data[..., 0]
    y = data[..., 1]

    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


def f3_grad(data: torch.Tensor) -> torch.Tensor:
    x = data[..., 0]
    y = data[..., 1]

    grad_x = -400 * x * (y - x ** 2) - 2 * (1 - x)
    grad_y = 200 * (y - x ** 2)

    return torch.stack([grad_x, grad_y], dim=-1)
