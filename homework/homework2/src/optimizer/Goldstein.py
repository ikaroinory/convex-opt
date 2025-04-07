from optimizer import Optimizer


class Goldstein(Optimizer):
    def __init__(self, f):
        super(Goldstein, self).__init__(f)

    def optimizer(self,x):
