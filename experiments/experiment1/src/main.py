import numpy as np
import torch
from torch import nn

from functions import rosenbrock_banana
from optimizer import *

torch.manual_seed(42)
torch.set_printoptions(precision=2)

f = lambda x: rosenbrock_banana(x, a=1, b=-100)

task_list = [
    {
        'task_name': 'Random Search',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': RandomSearch,
        'args': {
            'lr': 0.1
        },
        'epochs': 2500
    },
    {
        'task_name': 'Gradient Descent',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': GradientDescent,
        'args': {
            'lr': 1e-3
        },
        'epochs': 11000
    },
    {
        'task_name': 'Subgradient Descent',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': SubgradientDescent,
        'args': {
            'lr': 1e-3
        },
        'epochs': 11000
    },
    {
        'task_name': 'Conjugate Direction',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': ConjugateDirection,
        'args': {
            'lr': 1e-3,
            'beta_type': 'PR'
        },
        'epochs': 25000
    },
    {
        'task_name': 'Conjugate Gradient',
        'x': nn.Parameter(torch.tensor([1.5, 2.])),
        'optimizer': ConjugateGradient,
        'args': {
            'lr': 1e-3
        },
        'epochs': 500
    },
    {
        'task_name': 'BFGS',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': BFGS,
        'args': {
            'lr': 1e-3
        },
        'epochs': 22500
    },
    {
        'task_name': 'SGD',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': StochasticGradientDescent,
        'args': {
            'lr': 1e-3,
            'momentum': 0.8
        },
        'epochs': 7500
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
        'epochs': 3000
    },
    {
        'task_name': 'Admm',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': Admm,
        'args': {
            'lr': 1e-3,
            'rho': 1000,
            'l1_weight': 0
        },
        'epochs': 23000
    },
    {
        'task_name': 'Krylov',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': Krylov,
        'args': {
            'lr': 1e-3,
            'beta': 0.9
        },
        'epochs': 7000
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

    optimizer.show_loss(title=task_name, save=True, padding_to=25000)
    optimizer.show_points(f, title=task_name, save=True)

    best_point = optimizer.point_list[np.argmin(optimizer.loss_list)]

    print(f'Optimizer: {task_name}')
    print(f'Best point: {best_point}')
    print(f'Min Value: {f(best_point):.2f}')
    print(f'Iterator count: {np.argmin(optimizer.loss_list)}')
    print()
