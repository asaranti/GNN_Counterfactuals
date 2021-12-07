"""
    Node Classifier's GCN:
    https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-11-11
"""

import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    """
    GCN for Node Classification
    """

    def __init__(self, num_node_features: int, hidden_channels: int, num_classes: int):
        """
        Init node classification

        :param num_node_features:
        :param hidden_channels:
        :param num_classes:
        """

        super(GCN, self).__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        """
        Forward function

        :param x:
        :param edge_index:
        """

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

