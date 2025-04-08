import torch

from optimizer.Optimizer import Optimizer


class ImprovedWolfePowell(Optimizer):
    def __init__(self, f, f_grad, x0, d0, rho=None, sigma=None, gamma=None):
        super(ImprovedWolfePowell, self).__init__(f, f_grad)

        self.x0 = x0
        self.d0 = d0

        self.rho = rho if rho is not None else 1e-4
        self.sigma = sigma if sigma is not None else 0.2

        self.gamma = gamma if gamma is not None else 0.5

    def optimize(self, alpha0):
        alpha = alpha0
        while True:
            left1 = self.f(self.x0 + alpha * self.d0)
            right1 = self.f(self.x0) + self.rho * alpha * self.grad_f(self.x0) @ self.d0.T
            left2 = torch.abs(self.grad_f(self.x0 + alpha * self.d0) @ self.d0.T)
            right2 = -self.sigma * self.grad_f(self.x0) @ self.d0.T
            if left1 <= right1 and left2 <= right2:
                break

            alpha *= self.gamma

            self.iterator_count += 1

        return alpha
