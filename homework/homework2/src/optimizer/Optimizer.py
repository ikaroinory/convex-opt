class Optimizer:
    def __init__(self, f, grad_f):
        self.f = f
        self.grad_f = grad_f

        self.iterator_count = 0

    def __call__(self, *args, **kwargs):
        pass
