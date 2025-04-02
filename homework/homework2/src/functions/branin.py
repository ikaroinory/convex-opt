import torch


def branin(data: torch.Tensor, a=1, b=5.1 / 4 / torch.pi ** 2, c=5 * torch.pi, r=6, s=10, t=1 / (8 * torch.pi)) -> torch.Tensor:
    """

    Branin function.

    :param data: The input tensor with shape of (*, 2).
    :param a: Parameter.
    :param b: Parameter.
    :param c: Parameter.
    :param r: Parameter.
    :param s: Parameter.
    :param t: Parameter.
    :return: Branin function value with shape of (*).
    """

    x = data[..., 0]
    y = data[..., 1]

    item1 = a * (y - b * x ** 2 + c * x - r) ** 2
    item2 = s * (1 - t) * torch.cos(x)

    return item1 + item2 + s


def branin_grad(data: torch.Tensor, a=1, b=5.1 / 4 / torch.pi ** 2, c=5 * torch.pi, r=6, s=10, t=1 / (8 * torch.pi)) -> torch.Tensor:
    """

    Gradient of Branin function.

    :param data: The input tensor with shape of (*, 2).
    :param a: Parameter.
    :param b: Parameter.
    :param c: Parameter.
    :param r: Parameter.
    :param s: Parameter.
    :param t: Parameter.
    :return: Gradient of Branin function with shape of (*, 2).
    """

    x = data[..., 0]
    y = data[..., 1]

    grad_x = 2 * a * (y - b * x ** 2 + c * x - r) * (-2 * b * x + c) - s * (1 - t) * torch.sin(x)
    grad_y = 2 * a * (y - b * x ** 2 + c * x - r)

    return torch.stack([grad_x, grad_y], dim=-1)
