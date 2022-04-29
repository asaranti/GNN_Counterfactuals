"""
    Format Transformation from the synthetic dataset to Pytorch
    From synthetic to Pytorch

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-04-27
"""

import os
import pickle
import sys

import numpy as np
import torch

from plots.graph_visualization import graph_viz_general
from preprocessing_files.format_transformations.format_transformation_synth_to_pytorch import import_synthetic_data
from preprocessing_files.format_transformations.format_transformation_pytorch_to_ui import transform_from_pytorch_to_ui

########################################################################################################################
# [1.] Transformation Experiment ::: From Synthetic to Pytorch_Graph ===================================================
########################################################################################################################
dataset_folder = os.path.join("data", "input", "Synthetic", "synthetic_orig")
nodes_features_file = "FEATURES_synthetic.txt"
edges_features_file = "NETWORK_synthetic.txt"
target_values_features_file = "TARGET_synthetic.txt"

synthetic_graph_list = import_synthetic_data(
    dataset_folder,
    nodes_features_file,
    edges_features_file,
    target_values_features_file
)

########################################################################################################################
# [2.] Statistics of the dataset =======================================================================================
########################################################################################################################
x_min_all = sys.float_info.max
x_max_all = sys.float_info.min

for graph_idx in range(len(synthetic_graph_list)):

    print(f"Graph index: {graph_idx}")
    graph = synthetic_graph_list[graph_idx]

    graph_x = graph.x.cpu().detach().numpy()
    if x_max_all < np.amax(graph_x):
        x_max_all = np.amax(graph_x)
    if x_min_all > np.amin(graph_x):
        x_min_all = np.amin(graph_x)

print(f"x_min: {x_min_all}, x_max: {x_max_all}")

########################################################################################################################
# [3.] Visualization of the dataset ====================================================================================
########################################################################################################################
for graph_idx in range(len(synthetic_graph_list)):

    # print(f"Graph index: {graph_idx}")
    graph = synthetic_graph_list[graph_idx]

    # [2.1.] Check that the topology of all graphs is equal ------------------------------------------------------------
    if graph_idx == 0:
        graph_edge_index_0 = graph.edge_index
    else:
        b_topologie_equal = torch.equal(graph_edge_index_0, graph.edge_index)
        assert b_topologie_equal, f"Is the topology equal? {b_topologie_equal}"

    # [2.2.] Check that the edges are not double there -----------------------------------------------------------------
    all_edges = graph.edge_index.detach().cpu().numpy()
    columns_nr = all_edges.shape[1]

    all_edges_sorted_list = []
    all_edges_sorted_set = set()
    for column_index in range(columns_nr):

        edge_component = list(all_edges[:, column_index])
        edge_component.sort()

        all_edges_sorted_list.append(edge_component)
        all_edges_sorted_set.add(tuple(edge_component))

    if len(all_edges_sorted_list) != len(all_edges_sorted_set):
        print("Non unique edges")

    # [2.3.] Actual visualization --------------------------------------------------------------------------------------
    # graph_viz_general(graph)

########################################################################################################################
# [4.] Store the Pytorch Dataset =======================================================================================
########################################################################################################################
dataset_pytorch_folder = os.path.join("data", "output", "Synthetic", "synthetic_pytorch")
with open(os.path.join(dataset_pytorch_folder, 'synthetic_pytorch.pkl'), 'wb') as f:
    pickle.dump(synthetic_graph_list, f)

########################################################################################################################
# [5.] From Pytorch_Graph to UI Format =================================================================================
########################################################################################################################
dataset_ui_folder = os.path.join("data", "output", "Synthetic", "synthetic_ui")

graphs_nr = len(synthetic_graph_list)
graph_idx = 0
for graph_idx in range(graphs_nr):

    transform_from_pytorch_to_ui(synthetic_graph_list[graph_idx],
                                 dataset_ui_folder,
                                 f"synthetic_nodes_ui_format_{graph_idx}.csv",
                                 f"synthetic_egdes_ui_format_{graph_idx}.csv")

    graph_idx += 1

