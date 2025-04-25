from torch import nn


class OutputLayer(nn.Module):
    def __init__(self, in_num, layer_num, dim, inter_num=512):
        super().__init__()

        modules = []

        for i in range(layer_num):
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, dim))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, dim))
                modules.append(nn.BatchNorm1d(dim))
                modules.append(nn.LeakyReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)

        return out
