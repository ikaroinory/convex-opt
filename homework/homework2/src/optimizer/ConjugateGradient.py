import torch

from optimizer.Optimizer import Optimizer


class ConjugateGradient(Optimizer):
    def __init__(self, f, grad_f, method='PR'):
        super(ConjugateGradient, self).__init__(f, grad_f)

        self.method = method

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

    @staticmethod
    def __armijo_line_search(f, grad_f, x, d, alpha0=1.0, c1=1e-4, rho=0.5):
        alpha = alpha0
        while f(x + alpha * d) > f(x) + c1 * alpha * grad_f(x) @ d.T:
            alpha *= rho
        return alpha

    def optimize(self, x0: torch.Tensor, max_iter=100, epsilon=1e-6):
        x = x0
        grad = self.grad_f(x)
        d = -grad

        for _ in range(max_iter):
            if torch.norm(grad, p=2) < epsilon:
                break

            # alpha = 1e-3
            alpha = self.__armijo_line_search(self.f, self.grad_f, x, d)

            x = x + alpha * d
            grad_new = self.grad_f(x)
            beta = self.get_beta(grad, grad_new)
            grad = grad_new
            d = -grad + beta * d

            self.iterator_count += 1

        return x
