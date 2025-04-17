import torch


def mc_cormick(data: torch.Tensor) -> torch.Tensor:
    """

    McCormick function.

    :param data: The input tensor with shape of (*, 2).
    :return: McCormick function value with shape of (*).
    """

    x = data[..., 0]
    y = data[..., 1]

    return torch.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1


def mc_cormick_grad(data: torch.Tensor) -> torch.Tensor:
    """

    McCormick function gradient.

    :param data: The input tensor with shape of (*, 2).
    :return: McCormick function gradient with shape of (*, 2).
    """

    x = data[..., 0]
    y = data[..., 1]

    grad_x = torch.cos(x + y) + 2 * (x - y) - 1.5
    grad_y = torch.cos(x + y) - 2 * (x - y) + 2.5

    return torch.stack([grad_x, grad_y], dim=-1)
