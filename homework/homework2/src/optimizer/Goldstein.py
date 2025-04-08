from optimizer.Optimizer import Optimizer


class Goldstein(Optimizer):
    def __init__(self, f, f_grad, x0, d0, rho=None, gamma=None):
        super(Goldstein, self).__init__(f, f_grad)

        self.x0 = x0
        self.d0 = d0

        self.rho = rho if rho is not None else 1e-4
        self.gamma = gamma if gamma is not None else 0.9

    def optimize(self, alpha0):
        alpha = alpha0
        while True:
            mid = self.f(self.x0 + alpha * self.d0)
            right = self.f(self.x0) + self.rho * alpha * self.grad_f(self.x0) @ self.d0.T
            left = self.f(self.x0) + (1 - self.rho) * alpha * self.grad_f(self.x0) @ self.d0.T

            if left <= mid <= right:
                break

            alpha *= self.gamma

            self.iterator_count += 1

        return alpha
