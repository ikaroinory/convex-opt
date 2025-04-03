import numpy as np
import torch

from optimizer import Optimizer


class ShubertPiyavskiiSearch(Optimizer):
    def __init__(self, f, l):
        super(ShubertPiyavskiiSearch, self).__init__(f)

        self.l = l

    def optimize(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        x_samples = torch.tensor([alpha, beta])
        f_samples = torch.tensor([self.f(alpha), self.f(beta)])

        for _ in range(self.max_iter):
            max_lower_bound = np.inf
            best_x = (alpha + beta) / 2

            for i in range(len(x_samples) - 1):
                x_left, x_right = x_samples[i], x_samples[i + 1]
                f_left, f_right = f_samples[i], f_samples[i + 1]

                # 计算可能的最优插入点
                x_new = (x_left + x_right) / 2 - (f_right - f_left) / (2 * self.l)

                # 计算下界函数的值
                g_x_new = torch.min(f_samples - self.l * torch.abs(x_samples - x_new))

                if g_x_new > max_lower_bound:
                    max_lower_bound = g_x_new
                    best_x = x_new

            # 计算新点的目标函数值
            f_new = self.f(best_x)

            # 更新采样点
            x_samples = torch.cat([x_samples, torch.tensor([best_x], dtype=torch.float32)])
            f_samples = torch.cat([f_samples, torch.tensor([f_new], dtype=torch.float32)])

            # 重新排序
            sorted_indices = torch.argsort(x_samples)
            x_samples = x_samples[sorted_indices]
            f_samples = f_samples[sorted_indices]

        min_idx = torch.argmin(f_samples)
        return torch.tensor(x_samples[min_idx])
