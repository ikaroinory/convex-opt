import torch

from optimizer.Optimizer import Optimizer


class DFP(Optimizer):
    def __init__(self, f, f_grad, exact_line_search, H0):
        super(DFP, self).__init__(f, f_grad)

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

            H = H + (delta_x.T @ delta_x) / (torch.norm(delta_x)) - (H @ delta_grad.T @ delta_grad @ H) / (delta_grad @ H @ delta_grad.T)

            x = x_new

        return x
