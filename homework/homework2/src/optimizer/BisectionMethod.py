import torch

from optimizer.Optimizer import Optimizer


class BisectionMethod(Optimizer):
    def __init__(self, f, f_grad=None, epsilon=None):
        super(BisectionMethod, self).__init__(f, f_grad, epsilon=epsilon)

    def optimize(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        while beta - alpha >= self.epsilon:
            x = (alpha + beta) / 2
            f_grad_x = self.grad_f(x) if self.grad_f else self.f(x) - self.f(x + self.epsilon ** 2)

            if f_grad_x == 0:
                break

            if f_grad_x > 0:
                beta = x
            else:
                alpha = x

            self.iterator_count += 1

        return (alpha + beta) / 2
