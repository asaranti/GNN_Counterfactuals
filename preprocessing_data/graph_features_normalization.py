"""
    Graph features normalization

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-04-27
"""

import sys

import numpy as np
import torch


def graph_features_normalization(graphs_dataset: list) -> list:
    """
    The graph's features are stored in the field "x" (mandatory) and "edge_attr" (Optional - can be None).
    To be processed by a GNN they both have to be in a range between [0, +1] or [-1, +1]

    :param graphs_dataset: List of input graphs
    """

    device = 'cuda:0'

    # [0.] Compute the minimum and maximum for the normalization -------------------------------------------------------
    x_min_all = sys.float_info.max
    x_max_all = sys.float_info.min
    edge_features_min_all = sys.float_info.max
    edge_features_max_all = sys.float_info.min

    for graph in graphs_dataset:

        # [1.] Collect min-max for node features -----------------------------------------------------------------------
        x_features_array = graph.x.cpu().detach().numpy()

        if x_max_all < np.amax(x_features_array):
            x_max_all = np.amax(x_features_array)
        if x_min_all > np.amin(x_features_array):
            x_min_all = np.amin(x_features_array)

        # [2.] Collect min-max the edge features -----------------------------------------------------------------------
        if graph.edge_attr is not None:

            edge_features_array = graph.edge_attr.cpu().detach().numpy()

            if edge_features_max_all < np.amax(edge_features_array):
                edge_features_max_all = np.amax(edge_features_array)
            if edge_features_min_all > np.amin(edge_features_array):
                edge_features_min_all = np.amin(edge_features_array)

    # [3.] Now normalize -----------------------------------------------------------------------------------------------
    normalized_graphs_dataset = []
    for graph in graphs_dataset:

        x_features_array = graph.x.cpu().detach().numpy()
        x_features_transformed = (x_features_array - x_min_all) / (x_max_all - x_min_all)

        graph.x = torch.tensor(x_features_transformed).to(dtype=torch.float32)                          # float32 ~~~~~~

        if graph.edge_attr is not None:

            edge_features_array = graph.edge_attr.cpu().detach().numpy()
            edge_features_transformed = (edge_features_array - edge_features_min_all) / \
                                        (edge_features_max_all - edge_features_min_all)
            graph.edge_attr = torch.tensor(edge_features_transformed).to(dtype=torch.float32)           # float32 ~~~~~~

        # Insert to the normalized graphs list -------------------------------------------------------------------------
        graph.to(device)
        normalized_graphs_dataset.append(graph)

    # [4.] Returned the list of the normalized graphs ------------------------------------------------------------------
    return normalized_graphs_dataset
