from fractions import Fraction

import numpy as np


class StandardForm:
    def __init__(self, c: np.ndarray | list, A: np.ndarray | list, b: np.ndarray | list, enable_fraction: bool = False):
        if isinstance(c, list):
            c = np.array(c)
        if isinstance(A, list):
            A = np.array(A)
        if isinstance(b, list):
            b = np.array(b)

        self.c = np.vectorize(Fraction)(c) if enable_fraction else c.astype(np.float64)
        self.A = np.vectorize(Fraction)(A) if enable_fraction else A.astype(np.float64)
        self.b = np.vectorize(Fraction)(b) if enable_fraction else b.astype(np.float64)

        self.m, self.n = self.A.shape

    @staticmethod
    def _parse_objective_function(x):
        terms = []
        for i, val in enumerate(x, start=1):
            if val < 0:
                if i == 1:
                    terms.append('-')
                else:
                    terms.append(' - ')
                val = -val
            else:
                if i > 1:
                    terms.append(' + ')

            if val == 1:
                terms.append(f'x_{i}')
            else:
                terms.append(f'{val}x_{i}')
        return ''.join(terms)

    def _parse_constraint(self):
        terms = []
        for row, b_value in zip(list(self.A), list(self.b)):
            terms.append(f'{self._parse_objective_function(row)} = {b_value}')
        return terms

    def __str__(self):
        constraint_list = self._parse_constraint()
        s = (f'max  {self._parse_objective_function(list(self.c))}\n'
             f's.t. {constraint_list[0]}\n')
        for constraint in constraint_list[1:]:
            s += f'     {constraint}\n'

        s += '     '
        for i in range(1, self.n + 1):
            if i > 1:
                s += ', '
            s += f'x_{i}'
        s += ' >= 0'
        return s

    def solve(self):
        x_B_index_list = [i for i in range(self.m)]
        c_B = self.c[x_B_index_list]
        Ab = np.hstack([self.A, self.b.reshape([self.m, 1])])

        r = np.array([np.inf for _ in range(self.n)])

        while not np.all(r <= 0):
            for i, index in enumerate(x_B_index_list):
                if Ab[i, index] == 1:
                    continue

                Ab[i] = Ab[i] / Ab[i, index]

                for j in range(self.m):
                    if j == i:
                        continue
                    Ab[j] = Ab[j] - Ab[j, index] * Ab[i]

            r = self.c - np.diag(np.repeat(c_B, self.n).reshape([self.m, self.n]).T @ Ab[:, :-1])

            if np.all(r <= 0):
                break

            in_index = np.argmax(r)

            if np.all(Ab[:, in_index] <= 0):
                return None, None

            theta = Ab[:, -1] / Ab[:, :-1][:, in_index]
            theta[theta < 0] = np.inf
            out_x_B_index_list_index = np.argmin(theta)

            x_B_index_list[out_x_B_index_list_index] = int(in_index)
            c_B = self.c[x_B_index_list]

        x_star = np.zeros([self.n], dtype=np.float64)
        x_star[x_B_index_list] = Ab[:, -1]
        return x_star, self.c @ x_star
