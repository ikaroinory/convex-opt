from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt


class Visual:
    def __init__(self):
        self.point_list = []
        self.loss_list = []

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['text.usetex'] = True

    def show_loss(self, *, title, save=False, padding_to: int = None):
        loss_list = self.loss_list
        if padding_to is not None:
            loss_list += [loss_list[-1] for _ in range(padding_to - len(loss_list))]

        plt.plot(loss_list)
        loc = np.argmin(loss_list)
        plt.plot(loc, loss_list[loc], marker='^', color='red', markersize=10, label='Marked Point')

        plt.xlabel('Iteration')
        plt.ylabel('Value')

        plt.tight_layout()

        if save:
            Path('images').mkdir(parents=True, exist_ok=True)
            plt.savefig(f'images/{title}_loss.pdf')

        plt.show()

    def show_points(self, f, *, title, save=False):
        x = torch.linspace(-3, 3, 400)
        y = torch.linspace(-3, 3, 400)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        data = torch.stack([X, Y], dim=-1)
        Z = f(data)

        plt.figure(figsize=(6, 6))
        plt.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=30, cmap='viridis', linewidths=1.2)

        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.grid(True)

        length = np.argmin(self.loss_list) + 1
        path = torch.stack(self.point_list[:length], dim=0)

        plt.plot(path[:, 0], path[:, 1], color='red', linewidth=2, label='Path')
        plt.scatter(path[:, 0], path[:, 1], color='red', s=30)

        plt.tight_layout()

        if save:
            Path('images').mkdir(parents=True, exist_ok=True)
            plt.savefig(f'images/{title}_points.pdf')

        plt.show()
