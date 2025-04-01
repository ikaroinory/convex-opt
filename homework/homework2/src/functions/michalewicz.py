import torch


def michalewicz(data: torch.Tensor, m=10) -> torch.Tensor:
    """
    Michalewicz function.
    :param data: The input tensor with shape of (*, d).
    :param m: Parameter.
    :return: Michalewicz function value with shape of (*).
    """

    item = torch.sin(data) * torch.sin((torch.arange(1, data.shape[-1] + 1) * data ** 2) / torch.pi) ** (2 * m)

    return -torch.sum(item, dim=-1)
