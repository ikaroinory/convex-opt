import torch


def bohachevsky(data: torch.Tensor) -> torch.Tensor:
    """
    Bohachevsky function.
    :param data: The input tensor with shape of (*, 2).
    :return: Bohachevsky function value with shape of (*).
    """

    x = data[..., 0]
    y = data[..., 1]

    item1 = 0.3 * torch.cos(3 * torch.pi * x)
    item2 = 0.4 * torch.cos(4 * torch.pi * y)

    return x ** 2 + y ** 2 - item1 - item2 + 0.7
