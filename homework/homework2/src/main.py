import numpy as np
import torch

from function import f1, f1_grad, f2a
from functions import bohachevsky, bohachevsky_grad
from optimizer import ConjugateGradient, GoldenSection

np.set_printoptions(precision=4)

test1_list = [
    {
        'f': f1,
        'grad_f': f1_grad,
        'x0': torch.tensor([[0, 0]]).double()
    },
    {
        'f': bohachevsky,
        'grad_f': bohachevsky_grad,
        'x0': torch.tensor([[0, 0]]).double()
    }
]

test2_list = [{
    'f': f2a,
    'alpha': torch.tensor([-1]).double(),
    'beta': torch.tensor([1]).double()
}]

if __name__ == '__main__':
    for test in test1_list:
        f = test['f']
        grad_f = test['grad_f']
        x0 = test['x0']

        optimizer = ConjugateGradient(f, grad_f)
        x_star = optimizer.optimize(x0)

        print(f'f: {f.__name__}')
        print(f'Minimum point: {x_star.detach().numpy()}')
        print(f'Minimum value: {f(x_star).detach().numpy()}')
        print(f'Iteration count: {optimizer.iterator_count}')
        print()

    for test in test2_list:
        f = test['f']
        alpha = test['alpha']
        beta = test['beta']

        optimizer = GoldenSection(f, grad_f)
        x_star = optimizer.optimize(alpha, beta)

        print(f'f: {f.__name__}')
        print(f'Minimum point: {x_star}')
        print(f'Minimum value: {f(x_star)}')
        print(f'Iteration count: {optimizer.iterator_count}')
        print()
