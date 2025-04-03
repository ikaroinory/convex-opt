import torch

from optimizer.Optimizer import Optimizer


class GoldenSection(Optimizer):
    def __init__(self, f, epsilon=None):
        super(GoldenSection, self).__init__(f, epsilon=epsilon)

    def optimize(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        t = (torch.sqrt(torch.tensor(5)) - 1) / 2

        while beta - alpha >= self.epsilon:
            left = alpha + (1 - t) * (beta - alpha)
            right = alpha + t * (beta - alpha)

            if self.f(left) - self.f(right) > 0:
                alpha = left
            else:
                beta = right

            self.iterator_count += 1

        return (alpha + beta) / 2
