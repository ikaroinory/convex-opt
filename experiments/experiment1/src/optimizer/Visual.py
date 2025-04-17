from matplotlib import pyplot as plt


class Visual:
    def __init__(self):
        self.point_list = []
        self.loss_list = []

    def show(self, title):
        plt.plot(self.loss_list)
        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.show()
