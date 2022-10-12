"""
    GCN for classification
    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-10-05
"""

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    """
    GCN
    """

    def __init__(self, num_node_features: int, hidden_channels: int, num_classes: int):
        """
        Init
        :param num_node_features:
        :param hidden_channels:
        :param num_classes:
        """

        super(GCN, self).__init__()

        torch.manual_seed(12345)

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # self.conv4 = GCNConv(hidden_channels, hidden_channels)
        # self.conv5 = GCNConv(hidden_channels, hidden_channels)
        # self.conv6 = GCNConv(hidden_channels, hidden_channels)

        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        """
        Forward
        :param x:
        :param edge_index:
        :param batch:
        :param edge_weight:
        :return:
        """

        # 1. Obtain node embeddings ------------------------------------------------------------------------------------
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight)
        x = x.relu()
        # x = self.conv4(x, edge_index, edge_weight)
        # x = x.relu()
        # x = self.conv5(x, edge_index, edge_weight)
        # x = x.relu()
        # x = self.conv6(x, edge_index, edge_weight)

        # 2. Readout layer ---------------------------------------------------------------------------------------------
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier ----------------------------------------------------------------------------------
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin(x)

        """
        # 1. Obtain node embeddings ------------------------------------------------------------------------------------
        x = self.conv1(x, edge_index, edge_weight)
        x = F.sigmoid(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.sigmoid(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = F.sigmoid(x)
        x = self.conv4(x, edge_index, edge_weight)
        x = F.sigmoid(x)
        x = self.conv5(x, edge_index, edge_weight)
        x = F.sigmoid(x)
        x = self.conv6(x, edge_index, edge_weight)

        # 2. Readout layer ---------------------------------------------------------------------------------------------
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier ----------------------------------------------------------------------------------
        x = F.log_softmax(x, dim=1)
        """

        return x
