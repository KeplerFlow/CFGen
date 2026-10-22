import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SpatioTemporalGCN(nn.Module):
    def __init__(self, feature_dim: int, temporal_dim: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.lin_spatial = nn.Linear(feature_dim, hidden_channels, bias=False)
        self.lin_temporal = nn.Linear(temporal_dim, hidden_channels)

        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x_t: torch.Tensor, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.lin_spatial(x_t) + self.lin_temporal(z)

        h = self.conv1(h, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h
