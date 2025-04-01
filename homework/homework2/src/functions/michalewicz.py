import torch


def michalewicz(x: torch.Tensor, m=10) -> torch.Tensor:
    """
    Michalewicz function.
    :param x: The input tensor with shape of (*, d).
    :param m: Parameter.
    :return: Michalewicz function value with shape of (*).
    """

    item = torch.sin(x) * torch.sin((torch.arange(1, x.shape[-1] + 1) * x ** 2) / torch.pi) ** (2 * m)

    return -torch.sum(item, dim=-1)
