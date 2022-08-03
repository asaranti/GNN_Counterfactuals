"""
    Barabasi-Albert graphs generator

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-12-17
"""

from networkx.generators.random_graphs import barabasi_albert_graph
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils.random import barabasi_albert_graph

from utils.graph_utilities import remove_duplicate_edges


def ba_graphs_gen(graphs_nr: int, nodes_nr: int, edges_per_node_nr: int, node_features_nr: int, edge_features_nr: int) \
        -> list:
    """
    Generate a prespecified number of Barabasi-Albert [BA] graphs
    with configurable parameters

    :param graphs_nr: Number of graphs to be generated
    :param nodes_nr: Number of nodes in each graph
    :param edges_per_node_nr: Number of edges to attach from a new node to existing nodes
    :param node_features_nr: Number of node features per node
    :param edge_features_nr: Number of edge features per edge

    :return: List of generated graphs
    """

    graphs_list = []

    for graph_idx in range(graphs_nr):

        ba_graph_pytorch_orig_edge_index = barabasi_albert_graph(nodes_nr, edges_per_node_nr)
        ba_graph_pytorch_edge_index = torch.tensor(remove_duplicate_edges(ba_graph_pytorch_orig_edge_index.
                                                                          detach().cpu().numpy()))
        edges_nr = ba_graph_pytorch_edge_index.size(dim=1)

        ba_graph_x = torch.tensor(np.random.rand(nodes_nr, node_features_nr))
        ba_graph_edge_attr = torch.tensor(np.random.rand(edges_nr, edge_features_nr))

        node_labels = np.array([f"node_label_{x}" for x in range(nodes_nr)])
        node_ids = np.array([f"node_id_{x}" for x in range(nodes_nr)])
        node_feature_labels = np.array([f"node_feature_{x}" for x in range(node_features_nr)])
        edge_ids = np.array([f"edge_id_{x}" for x in range(edges_nr)])
        edge_attr_labels = np.array([f"edge_attr_{x}" for x in range(edge_features_nr)])

        protein_graph = Data(x=ba_graph_x, edge_index=ba_graph_pytorch_edge_index, edge_attr=ba_graph_edge_attr,
                             y=None, pos=None,
                             node_labels=node_labels,
                             node_ids=node_ids,
                             node_feature_labels=node_feature_labels,
                             edge_ids=edge_ids,
                             edge_attr_labels=edge_attr_labels,
                             graph_id=f"graph_id_{graph_idx}_0")

        graphs_list.append(protein_graph)

    return graphs_list
