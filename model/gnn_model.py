# model/gnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data


class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # x_j has shape [E, in_channels]
        return self.relu(self.lin(x_j))


class GNNModel(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_layers=3):
        super(GNNModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GNNLayer(num_node_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GNNLayer(hidden_dim, hidden_dim))
        self.fc = nn.Linear(hidden_dim, 1)  # Predict energy

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        out = self.fc(x)
        return out
