import torch


def himmelblau(data: torch.Tensor) -> torch.Tensor:
    """

    Himmelblau function.

    :param data: The input tensor with shape of (*, 2).
    :return: Himmelblau function value with shape of (*).
    """

    x = data[..., 0]
    y = data[..., 1]

    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def himmelblau_grad(data: torch.Tensor) -> torch.Tensor:
    """

    Compute the gradient of Himmelblau function.

    :param data: The input tensor with shape of (*, 2).
    :return: The gradient tensor with shape of (*, 2).
    """

    x = data[..., 0]
    y = data[..., 1]

    grad_x = 2 * (x ** 2 + y - 11) * 2 * x + 2 * (x + y ** 2 - 7)
    grad_y = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * 2 * y

    return torch.stack([grad_x, grad_y], dim=-1)
