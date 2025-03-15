import linear_programming as lp

question_list = [
    (
        [6, 14, 13, 0, 0],
        [
            [1, 4, 2, 1, 0],
            [1, 2, 4, 0, 1]
        ],
        [48, 60]
    ),
    (
        [-3, 2, 4, 0, 0],
        [
            [4, 5, -2, 1, 0],
            [1, -2, 1, 0, 1]
        ],
        [22, 30]
    ),
    (
        [1, 1, 1, 0, 0, 0],
        [
            [-1, 0, -1, 1, 0, 0],
            [2, -3, 1, 0, 1, 0],
            [2, -5, 6, 0, 0, 1]
        ],
        [5, 3, 5]
    ),
    (
        [4, 2, 8, 0],
        [
            [2, -1, 3, 1],
            [1, 2, 4, 0]
        ],
        [30, 40]
    ),
    (
        [1, 1, 0],
        [[1, 1, 1]],
        [2]
    )
]

for i, j, k in question_list:
    standard_form = lp.StandardForm(i, j, k)
    print(standard_form)
    x_star, value = standard_form.solve()
    print(x_star)
    print(value)
