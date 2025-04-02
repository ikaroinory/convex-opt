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
