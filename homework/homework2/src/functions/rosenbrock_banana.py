import torch


def rosenbrock_banana(data: torch.Tensor, a=1, b=5) -> torch.Tensor:
    """
    Rosenbrock Banana function.
    :param data: The input tensor with shape of (*, 2).
    :param a: Parameter.
    :param b: Parameter.
    :return: Rosenbrock Banana function value with shape of (*).
    """

    x = data[..., 0]
    y = data[..., 1]

    return (a - x) ** 2 - b * (y - x ** 2) ** 2
