import torch


def himmelblau(data: torch.Tensor) -> torch.Tensor:
    """

    Himmelblau function.

    :param data: The input tensor with shape of (*, 2).
    :return: Himmelblau function value with shape of (*).
    """

    x = data[..., 0]
    y = data[..., 1]

    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
