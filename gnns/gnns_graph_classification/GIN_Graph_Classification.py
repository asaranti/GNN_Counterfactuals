"""
    GIN for classification

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-06-15
"""

import torch
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool


class GIN(torch.nn.Module):
    """
    GIN
    """

    def __init__(self, num_node_features: int, dim_h: int, num_classes: int):
        """
        Init

        :param num_node_features:
        :param dim_h:
        :param num_classes:
        """

        super(GIN, self).__init__()

        self.conv1 = GINConv(
            Sequential(Linear(num_node_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv4 = GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))

        self.lin1 = Linear(dim_h * 6, dim_h * 6)
        self.lin2 = Linear(dim_h * 6, dim_h * 6)
        self.lin3 = Linear(dim_h * 6, dim_h * 6)
        self.lin4 = Linear(dim_h * 6, dim_h * 6)
        self.lin6 = Linear(dim_h * 6, num_classes)

    def forward(self, x, edge_index, batch):
        """

        :param x:
        :param edge_index:
        :param batch:
        """

        # Node embeddings ----------------------------------------------------------------------------------------------
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        h4 = self.conv4(h3, edge_index)
        h5 = self.conv4(h4, edge_index)
        h6 = self.conv4(h5, edge_index) 

        # Graph-level readout ------------------------------------------------------------------------------------------
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        h4 = global_add_pool(h4, batch)
        h5 = global_add_pool(h5, batch)
        h6 = global_add_pool(h6, batch)

        # Concatenate graph embeddings ---------------------------------------------------------------------------------
        h = torch.cat((h1, h2, h3, h4, h5, h6), dim=1)

        # Classifier ---------------------------------------------------------------------------------------------------
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h, F.log_softmax(h, dim=1)
