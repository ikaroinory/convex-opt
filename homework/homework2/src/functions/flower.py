import torch


def flower(data: torch.Tensor, a=1, b=1, c=4) -> torch.Tensor:
    """

    Flower function.

    :param data: The input tensor with shape of (*, 2).
    :param a: Parameter.
    :param b: Parameter.
    :param c: Parameter.
    :return: Flower function value with shape of (*).
    """

    x = data[..., 0]
    y = data[..., 1]

    return a * torch.norm(data, p=2, dim=-1) + b * torch.sin(c * torch.arctan2(y, x))
