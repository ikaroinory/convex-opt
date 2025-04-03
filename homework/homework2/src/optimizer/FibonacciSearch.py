import torch

from optimizer.Optimizer import Optimizer


class FibonacciSearch(Optimizer):
    def __init__(self, f):
        super(FibonacciSearch, self).__init__(f)

        self.max_iter = 100
        self.epsilon = 1e-6

    @staticmethod
    def _fibonacci_numbers(n):
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i - 1] + fib[i - 2])
        return fib

    def optimize(self, alpha: torch.Tensor, beta: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
        n = 0
        fib = [1, 1]
        while fib[-1] < (beta - alpha) / epsilon:
            fib.append(fib[-1] + fib[-2])
            n += 1

        while n > 0:
            left = alpha + (beta - alpha) * fib[n - 1] / fib[n + 1]
            right = alpha + (beta - alpha) * fib[n] / fib[n + 1]

            if self.f(left) - self.f(right) > 0:
                alpha = left
            else:
                beta = right

            n -= 1

        return (alpha + beta) / 2
