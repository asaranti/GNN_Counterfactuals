"""
    Format Transformation
    From Random_Kirc to Pytorch

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-01-27
"""

import os

import networkx as nx
from networkx.algorithms.components import connected_components
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx


def import_random_kirc_data(input_dataset_folder: str,
                            pytorch_random_kirc_mRNA_attribute_file: str,
                            pytorch_random_kirc_methy_attribute_file: str,
                            pytorch_random_kirc_edges_file: str,
                            pytorch_random_kirc_label_file: str) -> list:
    """
    Import the data from random_kirc dataset and transform them to the pytorch format

    :param input_dataset_folder: Folder where data lies
    :param pytorch_random_kirc_mRNA_attribute_file: File containing the mRNA feature values. The number of rows
    denotes the graph and the columns represent the nodes
    :param pytorch_random_kirc_methy_attribute_file: File containing the methy feature values. The number of rows
    denotes the graph and the columns represent the nodes
    :param pytorch_random_kirc_edges_file: File containing the edges between nodes. It is the same for each graph
    :param pytorch_random_kirc_label_file: File containing the label of each graph (used in a graph classification
    task)

    :return: List of all graphs
    """

    ####################################################################################################################
    # [1.] mRNA and Methy attributes ===================================================================================
    ####################################################################################################################
    node_feature_labels = ["mRNA", "methy"]

    pytorch_random_kirc_mRNA_attribute_file = os.path.join(input_dataset_folder,
                                                           pytorch_random_kirc_mRNA_attribute_file)
    node_attribute_mRNA_orig = pd.read_csv(pytorch_random_kirc_mRNA_attribute_file, sep=' ')
    mRNA_nan_values = node_attribute_mRNA_orig.isnull().sum().sum()
    assert mRNA_nan_values == 0, f"The number of NaN values in the features must be 0, " \
                                 f"instead it is: {mRNA_nan_values}"

    pytorch_random_kirc_methy_attribute_file = os.path.join(input_dataset_folder,
                                                            pytorch_random_kirc_methy_attribute_file)
    node_attribute_methy_orig = pd.read_csv(pytorch_random_kirc_methy_attribute_file, sep=' ')

    # Check and deal with NaNs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    methy_nan_values = node_attribute_methy_orig.isnull().sum().sum()
    # assert methy_nan_values == 0, f"The number of NaN values in the features must be 0, " \
    #                              f"instead it is: {methy_nan_values}"

    # node_attribute_methy_orig = node_attribute_methy_orig.fillna(node_attribute_methy_orig.mean())

    column_has_nan = node_attribute_methy_orig.columns[node_attribute_methy_orig.isna().any()].tolist()
    columns_with_nan = node_attribute_methy_orig[column_has_nan]
    node_attribute_methy_processed = node_attribute_methy_orig.drop(columns_with_nan, axis=1)
    node_attribute_mRNA_processed = node_attribute_mRNA_orig.drop(columns_with_nan, axis=1)

    # Check if the node names in the two pandas dataframes are equal ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    node_names_mRNA_attribute_list = list(node_attribute_mRNA_processed.columns)
    node_names_methy_attribute_list = list(node_attribute_methy_processed.columns)
    node_names_same = node_names_mRNA_attribute_list == node_names_methy_attribute_list
    assert node_names_same, f"Node names must be equal for both node attribute files: {node_names_same}"

    # Check if the graph ids in the two pandas dataframes are equal ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    graph_ids_mRNA_list = node_attribute_mRNA_processed.index.tolist()
    graph_ids_methy_list = node_attribute_mRNA_processed.index.tolist()

    graph_ids_same = graph_ids_mRNA_list == graph_ids_methy_list
    assert graph_ids_same, f"Graph ids must be equal for both node attribute files: {graph_ids_same}"

    # Create the nodes attributes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    nr_of_nodes = len(node_attribute_mRNA_processed.columns)
    nr_of_graphs = len(node_attribute_mRNA_processed.index)
    node_attributes_list = []
    for row_nr in range(nr_of_graphs):
        feature_mRNA = node_attribute_mRNA_processed.iloc[row_nr].values
        feature_methy = node_attribute_methy_processed.iloc[row_nr].values

        node_attributes = np.vstack((feature_mRNA.T, feature_methy.T)).T
        node_attributes_list.append(torch.tensor(node_attributes, dtype=torch.float32))

    ####################################################################################################################
    # [2.] Edges =======================================================================================================
    ####################################################################################################################

    # [2.1.]Create a dictiionary from node names to node numbering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    row_indexes_list = list(range(0, len(node_names_mRNA_attribute_list)))
    node_name_row_indexes_dict = dict(zip(node_names_mRNA_attribute_list, row_indexes_list))
    # print(node_name_row_indexes_dict)

    # [2.2.] Iterate over edges ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    line_cnt = 0
    random_kirc_edges = open(os.path.join(input_dataset_folder, pytorch_random_kirc_edges_file), "r")
    edges_left_indexes = []
    edges_right_indexes = []
    edge_attr = []
    edge_ids = []
    for line in random_kirc_edges:

        if line_cnt >= 1:

            line_array = line.split(' ')

            # [2.3.] Create the edge attributes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            combined_score = float(line_array[3].replace('\n', ''))
            if combined_score < 995:
                continue
            edge_attr.append(combined_score)

            # [2.4.] Create the edge indexes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node_left_name = line_array[1].replace('\"', '')
            node_right_name = line_array[2].replace('\"', '')

            if node_left_name in node_name_row_indexes_dict and node_right_name in node_name_row_indexes_dict:

                node_left_idx = node_name_row_indexes_dict[node_left_name]
                node_right_idx = node_name_row_indexes_dict[node_right_name]

                edges_left_indexes.append(node_left_idx)
                edges_right_indexes.append(node_right_idx)

                # [2.5.] Get the edge_ids ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                edge_ids.append(line_array[0])

        line_cnt += 1

    print(f"Nr. edges in orig: {len(edges_left_indexes)}")

    edge_idx = torch.tensor([edges_left_indexes, edges_right_indexes], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float64)

    ####################################################################################################################
    # [3.] Label of each graph =========================================================================================
    ####################################################################################################################
    pytorch_random_kirc_label_file = os.path.join(input_dataset_folder, pytorch_random_kirc_label_file)

    random_kirc_target_orig = pd.read_csv(pytorch_random_kirc_label_file, sep=' ')

    ####################################################################################################################
    # [4.] Graphs ======================================================================================================
    ####################################################################################################################
    node_ids_list = [f"node_id_{x}" for x in range(0, nr_of_nodes)]

    graph_all = []
    for row_nr in range(nr_of_graphs):

        # print(f"Graph Nr.: {row_nr}")

        graph_id = graph_ids_mRNA_list[row_nr]
        label = random_kirc_target_orig[graph_id].values[0]

        graph_orig = Data(
            x=node_attributes_list[row_nr],
            edge_index=edge_idx,
            edge_attr=edge_attr,
            y=torch.tensor([label]),
            pos=None,
            node_labels=np.array(node_names_mRNA_attribute_list),
            node_ids=np.array(node_ids_list),
            node_feature_labels=node_feature_labels,
            edge_ids=edge_ids,
            edge_attr_labels=["combined_score"],
            graph_id=graph_id
        )

        # Check if the graph is connected ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # graph_nx = to_networkx(graph, to_undirected=True)
        # graph_is_connected = nx.is_connected(graph_nx)
        # print(f"Graph is connected: {graph_is_connected}")

        # Compute the connected components and select the biggest graph ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        graph_nx = to_networkx(graph_orig, to_undirected=True)
        cc_graphs = connected_components(graph_nx)
        largest_cc_graph = list(max(cc_graphs, key=len))

        node_attributes_cc = node_attributes_list[row_nr][largest_cc_graph]

        nodes_indexes_list = list(range(0, len(largest_cc_graph)))
        node_reindexing_dict = dict(zip(largest_cc_graph, nodes_indexes_list))

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        edge_idx_array = edge_idx.cpu().detach().numpy()
        edge_idx_cc_left_list = []
        edge_idx_cc_right_list = []
        for edge_index in range(edge_idx_array.shape[1]):

            edge_left = int(edge_idx_array[0, edge_index])
            edge_right = int(edge_idx_array[1, edge_index])

            if edge_left in largest_cc_graph and edge_right in largest_cc_graph and \
                    edge_left in node_reindexing_dict and edge_right in node_reindexing_dict:
                edge_idx_cc_left_list.append(node_reindexing_dict[edge_left])
                edge_idx_cc_right_list.append(node_reindexing_dict[edge_right])

        edge_idx_cc = torch.tensor([edge_idx_cc_left_list, edge_idx_cc_right_list], dtype=torch.long)

        # print(edge_idx_cc)
        # print(f"Nr. edges in cc: {len(edge_idx_cc_left_list)}")

        graph_cc = Data(
            x=node_attributes_cc,
            edge_index=edge_idx_cc,
            y=torch.tensor([label]),
        )

        # print(graph_cc)
        graph_all.append(graph_cc)

    return graph_all
