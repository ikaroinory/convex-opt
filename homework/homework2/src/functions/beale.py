import torch


def beale(data: torch.Tensor) -> torch.Tensor:
    """
    Beale function.
    :param data: The input tensor with shape of (*, 2).
    :return: Beale function value with shape of (*).
    """

    x = data[..., 0]
    y = data[..., 1]

    item1 = (1.5 - x + x * y) ** 2
    item2 = (2.25 - x + x * y ** 2) ** 2
    item3 = (2.625 - x + x * y ** 3) ** 2

    return item1 + item2 + item3
