"""
    Format Transformation
    From Pytorch to UI
    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-12-06
"""

import csv
import os

import pandas as pd
from torch_geometric.data import Data


def transform_from_pytorch_to_ui(graph: Data):
    """
    Apply the transformation between the Pytorch format to UI format
    :param graph: Graph data
    :param input_dataset_folder: Dataset folder where the files lie
    :param ui_to_pytorch_nodes_file: Nodes file
    :param ui_to_pytorch_edges_file: Edges file
    """

    ####################################################################################################################
    # [1.] Nodes =======================================================================================================
    ####################################################################################################################
    node_attributes_df = pd.DataFrame(graph.x.detach().cpu().numpy(), columns=graph.node_feature_labels)

    node_attributes_df["label"] = graph.node_labels
    node_attributes_df["id"] = graph.node_ids.values()

    node_attributes_column_values = list(node_attributes_df.columns.values)
    del node_attributes_column_values[-2:]
    node_attributes_column_values = ['label', 'id'] + node_attributes_column_values

    node_attributes_df = node_attributes_df[node_attributes_column_values]
    # print(node_attributes_df.head(5))

    ####################################################################################################################
    # [2.] Edges =======================================================================================================
    ####################################################################################################################
    edge_index = graph.edge_index.detach().cpu().numpy()
    edge_from_col_vals = []
    edge_to_col_vals = []
    for col_nr in range(edge_index.shape[1]):

        edge = edge_index[:, col_nr]

        edge_from_col_vals.append(graph.node_ids[edge[0]])
        edge_to_col_vals.append(graph.node_ids[edge[1]])

    edge_data_dict = {"from": edge_from_col_vals, "to": edge_to_col_vals, "id": graph.edge_ids}
    edge_data_df = pd.DataFrame(data=edge_data_dict)

    edge_attributes_df = pd.DataFrame(data=graph.edge_attr.detach().cpu().numpy(),
                                      columns=graph.edge_attr_labels)
    edge_attributes_concat_df = pd.concat([edge_data_df, edge_attributes_df], axis=1)
    # print(edge_attributes_concat_df.head(5))

    return node_attributes_df, edge_attributes_concat_df