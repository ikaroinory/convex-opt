import torch


def wheeler(x: torch.Tensor, a=1.5) -> torch.Tensor:
    """
    Wheeler function.
    :param x: The input tensor with shape of (*, 2).
    :param a: Parameter.
    :return: Wheeler function value with shape of (*).
    """

    return -torch.exp(-(x[..., 0] * x[..., 1] - a) ** 2 - (x[..., 1] - a) ** 2)
