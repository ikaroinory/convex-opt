import torch


def beale(x: torch.Tensor) -> torch.Tensor:
    """
    Beale function.
    :param x: The input tensor with shape of (*, 2).
    :return: Beale function value with shape of (*).
    """

    item1 = (1.5 - x[..., 0] + x[..., 0] * x[..., 1]) ** 2
    item2 = (2.25 - x[..., 0] + x[..., 0] * x[..., 1] ** 2) ** 2
    item3 = (2.625 - x[..., 0] + x[..., 0] * x[..., 1] ** 3) ** 2

    return item1 + item2 + item3
