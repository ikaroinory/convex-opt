import torch

from function import f1, f1_grad, f2a, f2a_grad, f2b, f2b_grad, f3, f3_grad
from functions import bohachevsky, bohachevsky_grad
from optimizer import Armijo, BisectionSearch, ConjugateGradient, FibonacciSearch, GoldenSearch, ShubertPiyavskiiSearch

test_list = [
    [
        {
            'optimizer': ConjugateGradient,
            'init': {'f': f1, 'grad_f': f1_grad},
            'call': {'x0': torch.tensor([[0, 0]]).double()}
        },
        {
            'optimizer': ConjugateGradient,
            'init': {'f': bohachevsky, 'grad_f': bohachevsky_grad},
            'call': {'x0': torch.tensor([[0, 0]]).double()}
        }
    ],
    [
        {
            'optimizer': GoldenSearch,
            'init': {'f': f2a, 'epsilon': 0.06},
            'call': {'alpha': torch.tensor([-1]).double(), 'beta': torch.tensor([1]).double()}
        },
        {
            'optimizer': FibonacciSearch,
            'init': {'f': f2a, 'epsilon': 0.06},
            'call': {'alpha': torch.tensor([-1]).double(), 'beta': torch.tensor([1]).double()}
        },
        {
            'optimizer': BisectionSearch,
            'init': {'f': f2a, 'f_grad': f2a_grad, 'epsilon': 0.06},
            'call': {'alpha': torch.tensor([-1]).double(), 'beta': torch.tensor([1]).double()}
        },
        {
            'optimizer': ShubertPiyavskiiSearch,
            'init': {'f': f2a, 'l': 2, 'epsilon': 0.06},
            'call': {'alpha': torch.tensor([-1]).double(), 'beta': torch.tensor([1]).double()}
        },
        {
            'optimizer': GoldenSearch,
            'init': {'f': f2b, 'epsilon': 0.08},
            'call': {'alpha': torch.tensor([0]).double(), 'beta': torch.tensor([25]).double()}
        },
        {
            'optimizer': FibonacciSearch,
            'init': {'f': f2b, 'epsilon': 0.08},
            'call': {'alpha': torch.tensor([0]).double(), 'beta': torch.tensor([25]).double()}
        },
        {
            'optimizer': BisectionSearch,
            'init': {'f': f2b, 'f_grad': f2b_grad, 'epsilon': 0.08},
            'call': {'alpha': torch.tensor([0]).double(), 'beta': torch.tensor([25]).double()}
        },
        {
            'optimizer': ShubertPiyavskiiSearch,
            'init': {'f': f2b, 'l': 5, 'epsilon': 0.08},
            'call': {'alpha': torch.tensor([0]).double(), 'beta': torch.tensor([25]).double()}
        }
    ],
    [
        {
            'optimizer': Armijo,
            'init': {'f': f3, 'f_grad': f3_grad, 'x0': torch.tensor([[-1, 1]]).double(), 'd0': torch.tensor([[1, 1]]).double()},
            'call': {'alpha0': torch.tensor([[1]]).double()},
            'print_value': False
        },
    ]
]
