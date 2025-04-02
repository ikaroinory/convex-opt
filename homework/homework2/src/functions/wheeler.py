import torch


def wheeler(data: torch.Tensor, a=1.5) -> torch.Tensor:
    """

    Wheeler function.

    :param data: The input tensor with shape of (*, 2).
    :param a: Parameter.
    :return: Wheeler function value with shape of (*).
    """

    x = data[..., 0]
    y = data[..., 1]

    return -torch.exp(-(x * y - a) ** 2 - (y - a) ** 2)


def wheeler_grad(data: torch.Tensor, a=1.5) -> torch.Tensor:
    """

    Wheeler function gradient.

    :param data: The input tensor with shape of (*, 2).
    :param a: Parameter.
    :return: Wheeler function gradient with shape of (*, 2).
    """

    x = data[..., 0]
    y = data[..., 1]

    grad_x = 2 * (x * y - a) * y * torch.exp(-(x * y - a) ** 2 - (y - a) ** 2)
    grad_y = torch.exp(-(x * y - a) ** 2 - (y - a) ** 2) * (-2 * a * (1 + x) + 2 * y * (1 + x ** 2))

    return torch.stack([grad_x, grad_y], dim=-1)
