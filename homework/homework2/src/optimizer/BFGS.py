import torch

from .Optimizer import Optimizer


class BFGS(Optimizer):
    def __init__(self, f, f_grad, exact_line_search, H0):
        super(BFGS, self).__init__(f, f_grad)

        self.H0 = H0
        self.exact_line_search = exact_line_search

    def optimize(self, x0):
        x = x0
        H = self.H0
        for self.iterator_count in range(self.max_iter):
            grad = self.grad_f(x)
            if torch.norm(grad) < self.epsilon:
                break

            d = -grad @ H
            alpha = self.exact_line_search(x, d)
            x_new = x + alpha * d

            delta_x = x_new - x
            delta_grad = self.grad_f(x_new) - grad

            term1 = (1 + (delta_grad @ H @ delta_grad.T) / (delta_x @ delta_grad.T)) * ((delta_x.T @ delta_x) / (delta_x @ delta_grad.T))
            term2 = (H @ delta_grad.T @ delta_x + delta_x.T @ delta_grad @ H) / (delta_x @ delta_grad.T)
            H = H + term1 - term2

            x = x_new

        return x
