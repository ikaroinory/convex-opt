import torch

from optimizer.Optimizer import Optimizer


class GoldenSection(Optimizer):
    def __init__(self, f, grad_f=None, max_iter=100):
        super(GoldenSection, self).__init__(f, grad_f)

        self.max_iter = max_iter

    def optimize(self, alpha: torch.Tensor, beta: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
        t = (torch.sqrt(torch.tensor(5)) - 1) / 2

        a = alpha + (1 - t) * (beta - alpha)
        b = alpha + t * (beta - alpha)

        for self.iterator_count in range(self.max_iter):
            if beta - alpha < epsilon:
                break

            if self.f(a) - self.f(b) > 0:
                alpha = a
                a = b
                b = alpha + t * (beta - alpha)
            else:
                beta = b
                b = a
                a = alpha + (1 - t) * (beta - alpha)

        return (alpha + beta) / 2
