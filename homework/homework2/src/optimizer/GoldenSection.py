import torch

from optimizer.Optimizer import Optimizer


class GoldenSection(Optimizer):
    def __init__(self, f, max_iter=100):
        super(GoldenSection, self).__init__(f)

        self.max_iter = max_iter

    def optimize(self, alpha: torch.Tensor, beta: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
        t = (torch.sqrt(torch.tensor(5)) - 1) / 2

        left = alpha + (1 - t) * (beta - alpha)
        right = alpha + t * (beta - alpha)

        for self.iterator_count in range(self.max_iter):
            if beta - alpha < epsilon:
                break

            if self.f(left) - self.f(right) > 0:
                alpha = left
                left = right
                right = alpha + t * (beta - alpha)
            else:
                beta = right
                right = left
                left = alpha + (1 - t) * (beta - alpha)

        return (alpha + beta) / 2
