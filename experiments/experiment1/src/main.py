import numpy as np
import torch
from torch import nn

from functions import rosenbrock_banana
from optimizer import BFGS, ConjugateGradient, GradientDescent, Newton, StochasticGradientDescent, SubgradientDescent

f = lambda x: rosenbrock_banana(x, a=1, b=-100)

task_list = [
    {
        'task_name': 'Gradient Descent',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': GradientDescent,
        'args': {
            'lr': 1e-3
        },
        'epochs': 15000
    },
    {
        'task_name': 'Subgradient Descent',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': SubgradientDescent,
        'args': {
            'lr': 1e-3
        },
        'epochs': 15000
    },
    {
        'task_name': 'Conjugate Gradient',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': ConjugateGradient,
        'epochs': 45
    },
    {
        'task_name': 'BFGS',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': BFGS,
        'args': {
            'lr': 1
        },
        'epochs': 100
    },
    {
        'task_name': 'SGD',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': StochasticGradientDescent,
        'args': {
            'lr': 1e-3,
            'momentum': 0.8
        },
        'epochs': 10000
    },
    {
        'task_name': 'Newton',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': Newton,
        'epochs': 10
    }
]

for task in task_list:
    x = task['x']
    optimizer = task['optimizer']([x], **task.get('args', {}))

    for _ in range(task['epochs']):
        optimizer.zero_grad()
        loss = f(x)
        loss.backward()
        optimizer.step(lambda: f(x))
    optimizer.point_list.append(x.data)
    optimizer.loss_list.append(f(x).item())
    optimizer.show(task['task_name'])

    best_point = optimizer.point_list[np.argmin(optimizer.loss_list)]
    print(f'Best point: ({best_point[0]:>7.4f}, {best_point[1]:>7.4f})')
    print(f'Min Value: {np.min(optimizer.loss_list):>6.4f}')
    print(f'Iterator count: {np.argmin(optimizer.loss_list)}')
    print()
