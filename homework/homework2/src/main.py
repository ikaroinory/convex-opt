from test_list import test_list
from uilts import print_solution_information

if __name__ == '__main__':
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
