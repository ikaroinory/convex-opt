import torch


def circle(data: torch.Tensor) -> torch.Tensor:
    """
    Circle function.
    :param data: The input tensor with shape of (*, 2).
    :return: Circle function value with shape of (*, 2).
    """

    x = data[..., 0]
    y = data[..., 1]

    theta = x
    r = 0.5 + 0.5 * (2 * y) / (1 + y ** 2)

    return torch.stack([1 - r * torch.cos(theta), 1 - r * torch.sin(theta)], dim=-1)
