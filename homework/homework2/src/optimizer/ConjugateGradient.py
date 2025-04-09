import torch

from optimizer import Armijo
from optimizer.Optimizer import Optimizer


class ConjugateGradient(Optimizer):
    def __init__(self, f, grad_f, exact_line_search=None, method='PR', epsilon=None, max_iter=None):
        super(ConjugateGradient, self).__init__(f, grad_f, epsilon=epsilon, max_iter=max_iter)

        self.method = method
        self.exact_line_search = exact_line_search

    @staticmethod
    def __fletcher_reeves(grad_old: torch.Tensor, grad_new: torch.Tensor) -> torch.Tensor:
        return grad_new @ grad_new.T / grad_old @ grad_old.T

    @staticmethod
    def __polak_ribiere(grad_old: torch.Tensor, grad_new: torch.Tensor) -> torch.Tensor:
        return grad_new @ (grad_new - grad_old).T / grad_old @ grad_old.T

    @property
    def get_beta(self):
        if self.method == 'FR':
            return self.__fletcher_reeves
        if self.method == 'PR':
            return self.__polak_ribiere

    def optimize(self, x0: torch.Tensor):
        x = x0
        grad = self.grad_f(x)
        d = -grad

        for _ in range(self.max_iter):
            if torch.norm(grad) < self.epsilon:
                break

            if self.exact_line_search is None:
                optimizer = Armijo(self.f, self.grad_f, x, d)
                alpha = optimizer.optimize(torch.tensor([1], dtype=x.dtype, device=x.device))
            else:
                alpha = self.exact_line_search(x, d)

            x = x + alpha * d
            grad_new = self.grad_f(x)
            beta = self.get_beta(grad, grad_new)
            grad = grad_new
            d = -grad + beta * d

            self.iterator_count += 1

        return x
