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
