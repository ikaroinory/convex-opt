import torch.nn.functional as F
from torch import nn

from utils import get_device
from .GCNConv import GCNConv


class GCNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, out_dim):
        super().__init__()

        self.conv1 = GCNConv(in_channel, out_dim)
        self.conv2 = GCNConv(out_dim, out_channel)
        self.conv3 = GCNConv(in_channel, out_channel)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.bn2 = nn.BatchNorm1d(out_channel)

    def forward(self, x, edge_index):
        device = get_device()
        y = self.conv1(x, edge_index).to(device)

        y = F.relu(self.bn1(y))

        y = F.dropout(y, training=self.training)
        y = self.bn2(self.conv2(y, edge_index))

        return F.log_softmax(y, dim=1)
