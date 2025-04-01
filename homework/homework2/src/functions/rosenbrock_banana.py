import torch


def rosenbrock_banana(x: torch.Tensor, a=1, b=5) -> torch.Tensor:
    """
    Rosenbrock Banana function.
    :param x: The input tensor with shape of (*, 2).
    :param a: Parameter.
    :param b: Parameter.
    :return: Rosenbrock Banana function value with shape of (*).
    """

    item1 = (a - x[..., 0]) ** 2
    item2 = b * (x[..., 1] - x[..., 0] ** 2) ** 2

    return item1 - item2
