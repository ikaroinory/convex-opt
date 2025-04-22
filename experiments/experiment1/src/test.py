import torch
from torch import nn

from optimizer import *

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
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': ConjugateGradient,
        'args': {
            'lr': 1e-3
        },
        'epochs': 25000
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
        'task_name': 'ADMM',
        'x': nn.Parameter(torch.tensor([-1.5, 2.])),
        'optimizer': ADMM,
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
    print(r'\begin{figure}[ht]')
    print(r'    \centering')
    print(r'    \begin{subfigure}{0.4\textwidth}')
    print(r'        \centering')
    print(r'        \includegraphics[width=\textwidth]{figures/' + task['task_name'] + '_loss.pdf}')
    print(r'        \caption{最优值收敛曲线}')
    print(r'    \end{subfigure}')
    print(r'    \begin{subfigure}{0.4\textwidth}')
    print(r'        \centering')
    print(r'        \includegraphics[width=\textwidth]{figures/' + task['task_name'] + '_points.pdf}')
    print(r'        \caption{最优点收敛路径}')
    print(r'    \end{subfigure}')
    print(r'    \caption{' + task['task_name'] + '的最优值收敛曲线与最优点收敛路径}')
    print(r'\end{figure}')
