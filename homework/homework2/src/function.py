import torch


def f1(data: torch.Tensor) -> torch.Tensor:
    x = data[..., 0]
    y = data[..., 1]

    return x - y + 2 * x ** 2 + 2 * x * y + y ** 2


def f1_grad(data: torch.Tensor) -> torch.Tensor:
    x = data[..., 0]
    y = data[..., 1]

    return torch.stack([1 + 4 * x + 2 * y, -1 + 2 * x + 2 * y], dim=-1)


def f2a(data: torch.Tensor) -> torch.Tensor:
    x = data[..., 0]

    return 2 * x ** 2 - x - 1


def f2a_grad(data: torch.Tensor) -> torch.Tensor:
    x = data[..., 0]

    return 4 * x - 1


def f2b(data: torch.Tensor) -> torch.Tensor:
    x = data[..., 0]

    return 3 * x ** 2 - 21.6 * x - 1


def f2b_grad(data: torch.Tensor) -> torch.Tensor:
    x = data[..., 0]

    return 6 * x - 21.6


def f3(data: torch.Tensor) -> torch.Tensor:
    x = data[..., 0]
    y = data[..., 1]

    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


def f3_grad(data: torch.Tensor) -> torch.Tensor:
    x = data[..., 0]
    y = data[..., 1]

    x_grad = -400 * x * (y - x ** 2) - 2 * (1 - x)
    y_grad = 200 * (y - x ** 2)

    return torch.stack([x_grad, y_grad], dim=-1)


def f5(data: torch.Tensor) -> torch.Tensor:
    x = data[..., 0]
    y = data[..., 1]

    return 10 * x ** 2 + y ** 2


def f5_grad(data: torch.Tensor) -> torch.Tensor:
    x = data[..., 0]
    y = data[..., 1]

    return torch.stack([20 * x, 2 * y], dim=-1)


def f5_exact_line_search(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    x0 = x[..., 0]
    x1 = x[..., 1]

    d0 = d[..., 0]
    d1 = d[..., 1]

    return -(10 * x0 * d0 + x1 * d1) / (10 * d0 ** 2 + d1 ** 2)


def f6(data: torch.Tensor) -> torch.Tensor:
    x = data[..., 0]
    y = data[..., 1]

    return x ** 2 + 4 * y ** 2 - 4 * x - 8 * y


def f6_grad(data: torch.Tensor) -> torch.Tensor:
    x = data[..., 0]
    y = data[..., 1]

    return torch.stack([2 * x - 4, 8 * y - 8], dim=-1)


def f6_exact_line_search(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0]
    x2 = x[..., 1]

    d1 = d[..., 0]
    d2 = d[..., 1]

    return -(d1 * x1 + 4 * d2 * x2 - 2 * d1 - 4 * d2) / (d1 ** 2 + 4 * d2 ** 2 + 1e-6)
