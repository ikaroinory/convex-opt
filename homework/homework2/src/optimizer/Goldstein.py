from optimizer.Optimizer import Optimizer


class Goldstein(Optimizer):
    def __init__(self, f, f_grad, x0, d0, rho=None):
        super(Goldstein, self).__init__(f, f_grad)

        self.x0 = x0
        self.d0 = d0

        self.rho = rho if rho is not None else 0.25

    def optimize(self, alpha0):
        alpha = alpha0
        while True:
            if self.f(self.x0 + alpha * self.d0) <= self.f(self.x0) + self.rho * alpha * self.grad_f(self.x0) @ self.d0.T:
                break
            if self.f(self.x0 + alpha * self.d0) >= self.f(self.x0) + (1 - self.rho) * alpha * self.grad_f(self.x0) @ self.d0.T:
                break

            alpha *= self.rho

            self.iterator_count += 1

        return alpha
