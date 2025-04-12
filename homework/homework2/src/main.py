import torch
from torch.optim import SGD

from test_list import test_list
from uilts import print_solution_information


def q10():
    print(f'{"-" * 30} {10:02d} {"-" * 30}')

    def f(x):
        A = torch.tensor([[10, 0], [0, 1]], dtype=torch.float64, device='cuda')
        return x @ A @ x.T

    x = torch.randn([1, 2], dtype=torch.float64, requires_grad=True, device='cuda')

    optimizer = SGD([x], lr=0.001, momentum=0.9)

    best_x = x
    best_y = torch.inf
    best_iterator = 0
    for i in range(1000):
        optimizer.zero_grad()
        y = f(x)
        y.backward()
        optimizer.step()

        if y.item() < best_y:
            best_y = y.item()
            best_x = x.detach()
            best_iterator = i + 1

    print(f'Minimum point: {best_x}')
    print(f'Minimum value: {f(best_x)}')
    print(f'Iteration count: {best_iterator}')
    print('-' * 64)


if __name__ == '__main__':
    torch.manual_seed(42)

    for index, test in enumerate(test_list):
        if index > 0:
            print()

        print(f'{"-" * 30} {index + 1:02d} {"-" * 30}')
        print()
        for i in test:
            optimizer = i['optimizer'](**i['init'])
            x_star = optimizer.optimize(**i['call'])

            print_solution_information(optimizer, x_star, i.get('print_value', True))
        print('-' * 64)

    print()

    q10()
