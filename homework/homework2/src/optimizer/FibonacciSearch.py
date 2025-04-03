import torch

from optimizer.Optimizer import Optimizer


class FibonacciSearch(Optimizer):
    def __init__(self, f, epsilon=None):
        super(FibonacciSearch, self).__init__(f, epsilon=epsilon)

        self.max_iter = 100
        self.epsilon = 1e-6

    def optimize(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        n = 0
        fib = [1, 1]
        while fib[-1] < (beta - alpha) / self.epsilon:
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
            self.iterator_count += 1

        return (alpha + beta) / 2
