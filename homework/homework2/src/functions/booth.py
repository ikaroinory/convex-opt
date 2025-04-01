import torch


def booth(x: torch.Tensor) -> torch.Tensor:
    """
    Booth function.
    :param x: The input tensor with shape of (*, 2).
    :return: Booth function value with shape of (*).
    """

    return (x[..., 0] + 2 * x[..., 1] - 7) ** 2 + (2 * x[..., 0] + x[..., 1] - 5) ** 2
