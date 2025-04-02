import torch


def booth(data: torch.Tensor) -> torch.Tensor:
    """

    Booth function.

    :param data: The input tensor with shape of (*, 2).
    :return: Booth function value with shape of (*).
    """

    x = data[..., 0]
    y = data[..., 1]

    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
