import torch


def branin(x: torch.Tensor, a=1, b=5.1 / 4 / torch.pi ** 2, c=5 * torch.pi, r=6, s=10, t=1 / (8 * torch.pi)) -> torch.Tensor:
    """
    Branin function.
    :param x: The input tensor with shape of (*, 2).
    :param a: Parameter.
    :param b: Parameter.
    :param c: Parameter.
    :param r: Parameter.
    :param s: Parameter.
    :param t: Parameter.
    :return: Branin function value with shape of (*).
    """

    item1 = a * (x[..., 1] - b * x[..., 0] ** 2 + c * x[..., 0] - r) ** 2
    item2 = s * (1 - t) * torch.cos(x[..., 0])

    return item1 + item2 + s
