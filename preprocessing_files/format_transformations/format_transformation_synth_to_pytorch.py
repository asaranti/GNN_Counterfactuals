"""
    Format Transformation from the synthetic dataset to Pytorch
    From synthetic to Pytorch

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-04-27
"""

import os

import pandas as pd
import torch
from torch_geometric.data import Data
import uuid


def import_synthetic_data(input_dataset_folder: str,
                          nodes_features_file: str,
                          edges_features_file: str,
                          target_values_features_file: str):
    """
    Import Synthetic dataset

    :param input_dataset_folder: Folder where data lies
    :param nodes_features_file: Nodes features file name
    :param edges_features_file: Edges features file name
    :param target_values_features_file: File with target values of the graphs
    """

    ####################################################################################################################
    # [1.] Nodes features ==============================================================================================
    ####################################################################################################################
    nodes_features_file_abs = os.path.join(input_dataset_folder, nodes_features_file)
    nodes_features_orig = pd.read_csv(nodes_features_file_abs, sep=' ')
    nodes_features_nan_values = nodes_features_orig.isnull().sum().sum()
    assert nodes_features_nan_values == 0, f"The number of NaN values in the node features must be 0, " \
                                           f"instead it is: {nodes_features_nan_values}"

    ####################################################################################################################
    # [2.] Edges features ==============================================================================================
    ####################################################################################################################
    edges_features_file_abs = os.path.join(input_dataset_folder, edges_features_file)
    edges_features_orig = pd.read_csv(edges_features_file_abs, sep=' ')
    edges_features_nan_values = edges_features_orig.isnull().sum().sum()
    assert edges_features_nan_values == 0, f"The number of NaN values in the edge features must be 0, " \
                                           f"instead it is: {edges_features_nan_values}"

    ####################################################################################################################
    # [3.] Target values ===============================================================================================
    ####################################################################################################################
    target_values_file_abs = os.path.join(input_dataset_folder, target_values_features_file)
    target_values_orig = pd.read_csv(target_values_file_abs, sep=' ')
    target_values_nan_values = target_values_orig.isnull().sum().sum()
    assert target_values_nan_values == 0, f"The number of NaN values in the target values must be 0, " \
                                          f"instead it is: {target_values_nan_values}"

    ####################################################################################################################
    # [4.] Create the graphs ===========================================================================================
    ####################################################################################################################
    nodes_features_array = nodes_features_orig.values
    edges_features_array = edges_features_orig.values
    target_values_array = target_values_orig.values

    nodes_nr = nodes_features_array.shape[1]
    edges_nr = edges_features_array.shape[0]
    nodes_ids_list = [str(uuid.uuid4()) for node_index in range(nodes_nr)]
    nodes_labels_list = list(nodes_features_orig.columns)
    edge_ids_list = [f"edge_{edge_index}" for edge_index in range(edges_nr)]

    edge_idx_left_list = []
    edge_idx_right_list = []
    for edge_nr in range(edges_features_array.shape[0]):

        node_name_left = edges_features_array[edge_nr][0]
        edge_idx_left_list.append(nodes_labels_list.index(node_name_left))

        node_name_right = edges_features_array[edge_nr][1]
        edge_idx_right_list.append(nodes_labels_list.index(node_name_right))
    edge_idx = torch.tensor([edge_idx_left_list, edge_idx_right_list], dtype=torch.long)

    graphs_nr = target_values_array.shape[1]
    graph_all = []
    for graph_idx in range(graphs_nr):

        label = target_values_array[0][graph_idx]

        graph_all_nodes_features = nodes_features_array[graph_idx, :]
        graph_x = graph_all_nodes_features.reshape(graph_all_nodes_features.shape[0], -1)

        graph = Data(
            x=torch.from_numpy(graph_x).to(dtype=torch.float32),
            edge_index=edge_idx,
            edge_attr=None,
            y=torch.tensor([label]),
            pos=None,
            node_labels=nodes_labels_list,
            node_ids=nodes_ids_list,
            node_feature_labels=["node_feature_name"],
            edge_ids=edge_ids_list,
            edge_attr_labels=None,
            graph_id=f"graph_id_{graph_idx}_0"
        )

        graph_all.append(graph)

    return graph_all
