import torch


def ackley(data: torch.Tensor, a=20, b=0.2, c=2 * torch.pi) -> torch.Tensor:
    """

    Ackley function.

    :param a: Parameter.
    :param b: Parameter.
    :param c: Parameter.
    :param data: The input tensor with shape of (*, d).
    :return: Ackley function value with shape of (*).
    """

    item1 = -b * torch.sqrt(torch.sum(data ** 2, dim=-1) / data.shape[-1])
    item2 = torch.sum(torch.cos(c * data), dim=-1) / data.shape[-1]

    return -a * torch.exp(item1) - torch.exp(item2) + a + torch.e
