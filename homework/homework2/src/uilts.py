import torch

from optimizer import Optimizer


def print_solution_information(optimizer: Optimizer, x_star: torch.Tensor):
    print(f'Function: {optimizer.f.__name__}')
    print(f'Optimizer: {optimizer.__class__.__name__}')
    print(f'Minimum point: {x_star}')
    print(f'Minimum value: {torch.round(optimizer.f(x_star), decimals=4)}')
    print(f'Iteration count: {optimizer.iterator_count}')
    print()
