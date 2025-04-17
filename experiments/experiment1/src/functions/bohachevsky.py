import torch


def bohachevsky(data: torch.Tensor) -> torch.Tensor:
    """

    Bohachevsky function.

    :param data: The input tensor with shape of (*, 2).
    :return: Bohachevsky function value with shape of (*).
    """

    x = data[..., 0]
    y = data[..., 1]

    item1 = 0.3 * torch.cos(3 * torch.pi * x)
    item2 = 0.4 * torch.cos(4 * torch.pi * y)

    return x ** 2 + y ** 2 - item1 - item2 + 0.7


def bohachevsky_grad(data: torch.Tensor) -> torch.Tensor:
    """

    Gradient of Bohachevsky function.

    :param data: The input tensor with shape of (*, 2).
    :return: Gradient of Bohachevsky function with shape of (*, 2).
    """

    x = data[..., 0]
    y = data[..., 1]

    grad_x = 2 * x + 0.9 * torch.pi * torch.sin(3 * torch.pi * x)
    grad_y = 2 * y + 1.6 * torch.pi * torch.sin(4 * torch.pi * y)

    return torch.stack([grad_x, grad_y], dim=-1)
