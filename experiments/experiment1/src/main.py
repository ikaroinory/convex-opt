import numpy as np
import torch
from torch import nn

from functions import rosenbrock_banana
from optimizer import *

f = lambda x: rosenbrock_banana(x, a=1, b=-100)

task_list = [
    {
        'task_name': 'Random Search',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': RandomSearch,
        'args': {
            'lr': 0.1,
            'sigma': 0.1,
            'perturbation_scale': 1
        },
        'epochs': 5000
    },
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
        'task_name': 'Conjugate Direction',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': ConjugateDirection,
        'args': {
            'lr': 1e-3,
            'beta_type': 'FR'
        },
        'epochs': 47
    },
    {
        'task_name': 'Conjugate Gradient',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': ConjugateGradient,
        'args': {
            'lr': 1e-3
        },
        'epochs': 47
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
    },
    {
        'task_name': 'Damped Newton',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': DampedNewton,
        'args': {
            'lr': 2,
            'damping': 2
        },
        'epochs': 2000
    }
]

for task in task_list:
    task_name = task['task_name']
    x = task['x']
    epochs = task['epochs']

    optimizer = task['optimizer']([x], **task.get('args', {}))

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = f(x)
        loss.backward()
        optimizer.step(lambda: f(x))
    optimizer.point_list.append(x.data)
    optimizer.loss_list.append(f(x).item())
    optimizer.show(task_name)

    best_point = optimizer.point_list[np.argmin(optimizer.loss_list)]

    print(f'Optimizer: {task_name}')
    print(f'Best point: ({best_point[0]:>7.4f}, {best_point[1]:>7.4f})')
    print(f'Min Value: {np.min(optimizer.loss_list):>6.4f}')
    print(f'Iterator count: {np.argmin(optimizer.loss_list)}')
    print()
