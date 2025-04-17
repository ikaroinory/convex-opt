import torch


def booth(data: torch.Tensor) -> torch.Tensor:
    """

    Booth function.

    :param data: The input tensor with shape of (*, 2).
    :return: Booth function value with shape of (*).
    """

    x = data[..., 0]
    y = data[..., 1]

    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


def booth_grad(data: torch.Tensor) -> torch.Tensor:
    """

    Gradient of Booth function.

    :param data: The input tensor with shape of (*, 2).
    :return: The gradient tensor with shape of (*, 2).
    """

    x = data[..., 0]
    y = data[..., 1]

    grad_x = 10 * x + 8 * y - 34
    grad_y = 8 * x + 10 * y - 38

    return torch.stack([grad_x, grad_y], dim=-1)
