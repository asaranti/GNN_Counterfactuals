"""
    Format Transformation
    From UI to Pytorch

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-12-03
"""

import os

import pandas as pd
import torch
from torch_geometric.data import Data

from preprocessing.format_transformation_pytorch_to_ui import transform_from_pytorch_to_ui


def transform_from_ui_to_pytorch(input_dataset_folder: str,
                                 ui_to_pytorch_nodes_file: str,
                                 ui_to_pytorch_edges_file: str) -> Data:
    """
    Apply the transformation between UI format to the Pytorch format

    :param input_dataset_folder: Dataset folder where the files lie
    :param ui_to_pytorch_nodes_file: Nodes file
    :param ui_to_pytorch_edges_file: Edges file

    :return: Pytorch graph
    """

    ####################################################################################################################
    # [1.] Nodes =======================================================================================================
    ####################################################################################################################
    node_attributes_df = pd.read_csv(os.path.join(input_dataset_folder, ui_to_pytorch_nodes_file), sep=',')
    # print(node_attributes_df.head(5))

    # [1.1.] Create a mapping from node ids to row numbers -------------------------------------------------------------
    node_ids_list = node_attributes_df['id'].tolist()
    node_row_indexes_list = list(range(0, len(node_ids_list)))
    node_ids_row_indexes_dict = dict(zip(node_ids_list, node_row_indexes_list))
    # print(node_ids_row_indexes_dict)

    # [1.2.] Get the node labels and node attributes separately --------------------------------------------------------
    node_labels = node_attributes_df["label"].to_numpy()
    node_ids = node_attributes_df["id"].to_numpy()

    node_attributes_df.drop(["label", "id"], axis=1, inplace=True)
    # print(node_attributes_df.head(5))

    node_attributes_x = torch.tensor(node_attributes_df.to_numpy())
    node_feature_labels = node_attributes_df.columns.values

    ####################################################################################################################
    # [2.] Edges =======================================================================================================
    ####################################################################################################################
    edge_attributes_df = pd.read_csv(os.path.join(input_dataset_folder, ui_to_pytorch_edges_file), sep=',')

    edge_from = edge_attributes_df["from"].to_numpy()
    edge_to = edge_attributes_df["to"].to_numpy()
    edges_nr = len(edge_attributes_df.index)

    edge_left = []
    edge_right = []
    for edge_idx in range(edges_nr):
        edge_left.append(node_ids_row_indexes_dict[edge_from[edge_idx]])
        edge_right.append(node_ids_row_indexes_dict[edge_to[edge_idx]])
    edge_idx = torch.tensor([edge_left, edge_right])
    # print(edge_idx)

    edge_ids = edge_attributes_df["id"].to_numpy()

    edge_attributes_df.drop(["from", "to", "id"], axis=1, inplace=True)
    # print(edge_attributes_df.head(5))

    edge_attr_labels = edge_attributes_df.columns.values

    edge_attr = torch.tensor(edge_attributes_df.to_numpy())

    ####################################################################################################################
    # [3.] Graph =======================================================================================================
    ####################################################################################################################
    graph = Data(x=node_attributes_x, edge_index=edge_idx, edge_attr=edge_attr, y=None, pos=None,
                 node_labels=node_labels, node_ids=node_ids, node_feature_labels=node_feature_labels,
                 edge_ids=edge_ids, edge_attr_labels=edge_attr_labels)

    return graph

