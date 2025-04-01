import torch


def mc_cormick(data: torch.Tensor) -> torch.Tensor:
    """
    McCormick function.
    :param data: The input tensor with shape of (*, 2).
    :return: McCormick function value with shape of (*).
    """

    x = data[..., 0]
    y = data[..., 1]

    return torch.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1
