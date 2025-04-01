import torch


def flower(x: torch.Tensor, a=1, b=1, c=4) -> torch.Tensor:
    """
    Flower function.
    :param x: The input tensor with shape of (*, 2).
    :param a: Parameter.
    :param b: Parameter.
    :param c: Parameter.
    :return: Flower function value with shape of (*).
    """

    return a * torch.norm(x, p=2, dim=-1) + b * torch.sin(c * torch.arctan2(x[..., 1], x[..., 0]))
