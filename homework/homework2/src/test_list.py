import torch

from function import f1, f1_grad
from functions import bohachevsky, bohachevsky_grad
from optimizer import ConjugateGradient

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
    ]
]