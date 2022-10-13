"""
    GCN for classification
    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-10-05
"""

import os

import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    """
    GCN
    """

    def __init__(self,
                 num_node_features: int,
                 hidden_channels: int,
                 layers_nr: int,
                 num_classes: int,
                 ):
        """
        Init

        :param num_node_features: Number of node features (similar to the 3 RGB channels of images in CNNs)
        :param hidden_channels: Number of neurons in each layer
        :param layers_nr: Number of layers of the GNN
        :param num_classes: Number of output classes
        """

        super(GCN, self).__init__()

        torch.manual_seed(12345)

        self.hidden_channels = hidden_channels
        self.layers_nr = layers_nr

        self.conv_layers_list = []      # [1.] All layers in a list ----------------------------------------------------

        # [2.] Input layer ---------------------------------------------------------------------------------------------
        self.conv_layers_list.append(GCNConv(num_node_features, self.hidden_channels))

        # [3.] Intermediate layers -------------------------------------------------------------------------------------
        for intermediate_layer in range(self.layers_nr - 1):
            print(f"Intermediate Layer: {intermediate_layer}")
            self.conv_layers_list.append(GCNConv(self.hidden_channels, self.hidden_channels))

        self.gcns_modules = nn.ModuleList(self.conv_layers_list)

        # [4.] Last layer ----------------------------------------------------------------------------------------------
        self.lin = Linear(self.hidden_channels, num_classes)

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
        gcn_modules_len = len(self.gcns_modules)
        for gcn_module_idx in range(gcn_modules_len):

            gcn_module = self.gcns_modules[gcn_module_idx]

            if gcn_module_idx < gcn_modules_len - 1:
                x = gcn_module(x, edge_index, edge_weight)
                x = x.relu()
            else:
                x = gcn_module(x, edge_index, edge_weight)

        # 2. Readout layer ---------------------------------------------------------------------------------------------
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier ----------------------------------------------------------------------------------
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin(x)

        return x
