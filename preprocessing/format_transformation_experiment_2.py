"""
    Format Transformation Experiment Nr. 2

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-12-17
"""

import os

from networkx.generators.random_graphs import barabasi_albert_graph
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

from preprocessing.format_transformation_pytorch_to_ui import transform_from_pytorch_to_ui

########################################################################################################################
# [1.] BA graphs generation ============================================================================================
########################################################################################################################
graphs_nr = 100
nodes_nr = 10
edges_per_node_nr = 2    # Number of edges to attach from a new node to existing nodes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

node_features_nr = 5
edge_features_nr = 4

graphs_list = []
for graph_idx in range(graphs_nr):

    ba_graph_nx = barabasi_albert_graph(nodes_nr, edges_per_node_nr)
    ba_graph_pytorch_orig = from_networkx(ba_graph_nx)

    edges_nr = ba_graph_pytorch_orig.edge_index.size(dim=1)

    ba_graph_x = torch.tensor(np.random.rand(nodes_nr, node_features_nr))
    ba_graph_edge_attr = torch.tensor(np.random.rand(edges_nr, edge_features_nr))

    node_labels = np.array([f"node_label_{x}" for x in range(nodes_nr)])
    node_ids = np.array([f"node_id_{x}" for x in range(nodes_nr)])
    node_feature_labels = np.array([f"node_feature_{x}" for x in range(node_features_nr)])
    edge_ids = np.array([f"edge_id_{x}" for x in range(edges_nr)])
    edge_attr_labels = np.array([f"edge_attr_{x}" for x in range(edge_features_nr)])

    protein_graph = Data(x=ba_graph_x, edge_index=ba_graph_pytorch_orig.edge_index, edge_attr=ba_graph_edge_attr,
                         y=None, pos=None,
                         node_labels=node_labels,
                         node_ids=node_ids,
                         node_feature_labels=node_feature_labels,
                         edge_ids=edge_ids,
                         edge_attr_labels=edge_attr_labels,
                         graph_id=f"graph_ppi_{graph_idx}")

    graphs_list.append(protein_graph)

########################################################################################################################
# [2.] Pytorch to UI transformation ====================================================================================
########################################################################################################################
dataset_folder = os.path.join("data", "BA_Dataset")

graph_idx = 0
for graph_idx in range(graphs_nr):

    print(f"Transform the graph {graph_idx} to the UI format")

    transform_from_pytorch_to_ui(graphs_list[graph_idx],
                                 dataset_folder,
                                 f"ba_nodes_ui_format_{graph_idx}.csv",
                                 f"ba_edges_ui_format_{graph_idx}.csv")

    graph_idx += 1

