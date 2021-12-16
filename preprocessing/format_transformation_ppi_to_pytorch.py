"""
    Format Transformation
    From PPI to Pytorch

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-11-22
"""

from datetime import datetime
from collections import Counter
import os
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from plots.graph_statistics_plots import statistics_histogram


def import_whole_ppi_data(input_dataset_folder: str,
                          pytorch_to_ppi_attributes_file: str,
                          pytorch_to_ppi_node_id_to_name_file: str,
                          pytorch_to_ppi_edges_file: str):
    """
    Apply the transformation between PPI format to the Pytorch format to create the big graph

    :param input_dataset_folder: Dataset folder where the files lie
    :param pytorch_to_ppi_attributes_file: Attributes file
    :param pytorch_to_ppi_node_id_to_name_file: File containing the mapping from node ids to node names
    :param pytorch_to_ppi_edges_file: File containing the edge connections

    :return: Pytorch graph
    """

    ####################################################################################################################
    # [1.] Nodes =======================================================================================================
    ####################################################################################################################
    pytorch_to_ppi_attributes_file = open(os.path.join(input_dataset_folder, pytorch_to_ppi_attributes_file), "r")

    node_attributes_orig = pd.read_csv(pytorch_to_ppi_attributes_file, sep='\t')

    # Not mandatory: remove "attribute_name" column before GNN training ------------------------------------------------
    node_attributes_drop_attr_name_df = node_attributes_orig.drop(['attrib_name'], axis=1)
    # print(node_attributes_drop_attr_name_df.head(5))
    node_feature_labels = node_attributes_drop_attr_name_df.columns.values
    node_attributes = torch.tensor(node_attributes_drop_attr_name_df.values)
    # print(node_attributes_orig.shape, len(node_feature_labels))

    pytorch_to_ppi_attributes_file.close()

    # Attributes names dictionary correspondence with row --------------------------------------------------------------
    attrib_name_list = node_attributes_orig['attrib_name'].tolist()
    row_indexes_list = list(range(0, len(attrib_name_list)))
    attrib_name_row_indexes_dict = dict(zip(attrib_name_list, row_indexes_list))
    # print(len(attrib_name_row_indexes_dict))

    ####################################################################################################################
    # [2.] Mapping for edges ===========================================================================================
    ####################################################################################################################
    human_names = open(os.path.join(input_dataset_folder, pytorch_to_ppi_node_id_to_name_file), "r")
    human_names_proteins = {}
    line_cnt = 0
    for line in human_names:
        line_array = line.split('\t')

        if line_cnt >= 1:
            protein_name_2 = line_array[2].rstrip('\n')
            human_names_proteins[protein_name_2] = line_array[1]

        line_cnt += 1

    # print(len(human_names_proteins))

    ####################################################################################################################
    # [3.] Edges =======================================================================================================
    ####################################################################################################################
    edge_idx_left = []
    edge_idx_right = []

    edge_attr = []
    edge_ids = []

    protein_links = open(os.path.join(input_dataset_folder, pytorch_to_ppi_edges_file), "r")
    line_cnt = 0
    edge_id = 0
    for line in protein_links:

        if line_cnt == 0:
            line_array = line.split(" ")
            edge_attr_labels = [line_array[2].rstrip("\n")]

        if line_cnt >= 1:

            line_array = line.split('\t')
            line_edges_array = line.split(" ")

            # 3.1. Create the edge indexes =============================================================================
            protein_left_name = line_edges_array[0]
            protein_right_name = line_edges_array[1]

            # Protein left and protein right must have "human readable" names. If not the edge will not be added. ------
            protein_left_name_has_human_name = protein_left_name in human_names_proteins.keys()
            protein_right_name_has_human_name = protein_right_name in human_names_proteins.keys()

            if protein_left_name_has_human_name and protein_right_name_has_human_name:

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Using the Dataframe lasts longer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # protein_left_indexes_list = node_attributes_orig.index[node_attributes_orig['attrib_name'] ==
                #                                                       human_names_proteins[protein_left_name]].tolist()
                # protein_right_indexes_list = node_attributes_orig.index[node_attributes_orig['attrib_name'] ==
                #                                                        human_names_proteins[protein_right_name]].tolist()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                if human_names_proteins[protein_left_name] in attrib_name_row_indexes_dict.keys() and \
                        human_names_proteins[protein_right_name] in attrib_name_row_indexes_dict.keys():
                    protein_left_idx = attrib_name_row_indexes_dict[human_names_proteins[protein_left_name]]
                    protein_right_idx = attrib_name_row_indexes_dict[human_names_proteins[protein_right_name]]

                    # if len(protein_left_indexes_list) == 1 and len(protein_right_indexes_list) == 1:

                    # protein_left_idx = protein_left_indexes_list[0]
                    # protein_right_idx = protein_right_indexes_list[0]

                    edge_idx_left.append(protein_left_idx)
                    edge_idx_right.append(protein_right_idx)

                    # 3.2. Create the edge attributes ==================================================================
                    edge_attr.append([float(line_edges_array[2].rstrip('\n'))])

                    # 3.3. Append the edge_id ==========================================================================
                    edge_ids.append(f"edge_id_{edge_id}")
                    edge_id += 1
            # else:
            #    if not protein_left_name_has_human_name:
            #        print(f"Protein's name: {protein_left_name} not found in {pytorch_to_ppi_node_id_to_name_file}.")
            #    if not protein_right_name_has_human_name:
            #        print(f"Protein's name : {protein_left_name} not found in {pytorch_to_ppi_node_id_to_name_file}.")

        line_cnt += 1

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # print(f"edge_idx: {[edge_idx_left, edge_idx_right]}")
    # print(f"edge_attr: {edge_attr}")

    edge_idx = torch.tensor([edge_idx_left, edge_idx_right], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float64)

    ####################################################################################################################
    # [4.] Graphs ======================================================================================================
    ####################################################################################################################
    # [4.1.] Create a graph that encomapasses all elements that are imported -------------------------------------------
    node_ids_list = [f"node_id_{x}" for x in range(0, len(attrib_name_list))]
    protein_graph_all = Data(x=node_attributes, edge_index=edge_idx, edge_attr=edge_attr, y=None, pos=None,
                             node_labels=np.array(attrib_name_list),
                             node_ids=np.array(node_ids_list),
                             node_feature_labels=node_feature_labels,
                             edge_ids=edge_ids,
                             edge_attr_labels=edge_attr_labels,
                             graph_id="graph_ppi_all")

    return protein_graph_all


def transform_from_ppi_to_pytorch(input_dataset_folder: str,
                                  pytorch_to_ppi_attributes_file: str,
                                  pytorch_to_ppi_node_id_to_name_file: str,
                                  pytorch_to_ppi_edges_file: str) -> Data:
    """
    Apply the transformation between PPI format to the Pytorch format

    :param input_dataset_folder: Dataset folder where the files lie
    :param pytorch_to_ppi_attributes_file: Attributes file
    :param pytorch_to_ppi_node_id_to_name_file: File containing the mapping from node ids to node names
    :param pytorch_to_ppi_edges_file: File containing the edge connections

    :return: Pytorch graph
    """

    ####################################################################################################################
    # [0.] Import the whole (disconnected) graph and compute the connected components ==================================
    ####################################################################################################################
    datetime_1 = datetime.now()

    protein_graph_all = import_whole_ppi_data(input_dataset_folder,
                                              pytorch_to_ppi_attributes_file,
                                              pytorch_to_ppi_node_id_to_name_file,
                                              pytorch_to_ppi_edges_file)

    # [0.1.] Check if the graph is connected or can be split to several connected component ----------------------------
    protein_graph_all_networkx = to_networkx(protein_graph_all, to_undirected=True)
    # is_graph_connected = nx.is_connected(protein_graph_all_networkx)
    # print(f"Is the whole PPI graph connected? {is_graph_connected}")

    # [0.2.] Compute the adjacency matrix and then use this to the detection of connected components -------------------
    adj_matrix_all = nx.adjacency_matrix(protein_graph_all_networkx)
    n_graph_components, graph_labels = connected_components(csgraph=adj_matrix_all,
                                                            directed=False,
                                                            return_labels=True)
    # print(f"Nr. of graph components: {n_graph_components}")
    # print(f"Graph labels: {len(graph_labels)}")

    ####################################################################################################################
    # [1.] Create the individual graphs ================================================================================
    ####################################################################################################################
    node_sizes = []
    graph_idx = 0
    protein_graph_list = []
    for graph_component_idx in range(n_graph_components):

        # [1.1.] Return all indexes of the nodes that belong to the N-th component -------------------------------------
        node_indexes = [i for i, x in enumerate(graph_labels) if x == graph_component_idx]
        node_sizes.append(len(node_indexes))

        x_cc = protein_graph_all.x[node_indexes]  # x ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        row_indexes_list = list(range(0, len(node_indexes)))
        node_indexes_row_indexes_dict = dict(zip(node_indexes, row_indexes_list))

        if len(node_indexes) > 1:

            # [1.2.] Edges ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            edge_index_array_2d = protein_graph_all.edge_index.cpu().detach().numpy()
            edge_index_array_1d = edge_index_array_2d.reshape((1, 2*edge_index_array_2d.shape[1]), order='F')[0]

            indexes_of_cc_edge_elements = list(np.where(np.in1d(edge_index_array_1d, node_indexes))[0])
            edge_index_selected_1d = list(edge_index_array_1d[indexes_of_cc_edge_elements])
            edge_index_selected_1d = np.array([node_indexes_row_indexes_dict[key] for key in edge_index_selected_1d])

            edge_index_selected_2d = torch.tensor(edge_index_selected_1d.reshape(
                (2, int(edge_index_selected_1d.shape[0]/2)), order='F'))

            # [1.3.] Further parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            indexes_cc_edge_ids = indexes_of_cc_edge_elements[::2]
            indexes_cc_edge_ids = [int(x/2) for x in indexes_cc_edge_ids]

            edge_attr_cc = protein_graph_all.edge_attr[indexes_cc_edge_ids]     # edge_attr ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            edge_ids_cc = np.array(protein_graph_all.edge_ids)[indexes_cc_edge_ids]
            edge_attr_labels_cc = protein_graph_all.edge_attr_labels

        else:

            edge_index_selected_2d = torch.tensor(np.empty([2, 0]))
            edge_attr_cc = torch.tensor(np.empty([0, 1]))
            edge_ids_cc = np.empty([])
            edge_attr_labels_cc = protein_graph_all.edge_attr_labels

        node_labels_cc = protein_graph_all.node_labels[node_indexes]
        node_ids_cc = protein_graph_all.node_ids[node_indexes]
        node_feature_labels_cc = protein_graph_all.node_feature_labels

        # [1.4.] Create the graph and append it to the list of graph that will be returned ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        protein_graph = Data(x=x_cc, edge_index=edge_index_selected_2d, edge_attr=edge_attr_cc, y=None, pos=None,
                             node_labels=node_labels_cc,
                             node_ids=node_ids_cc,
                             node_feature_labels=node_feature_labels_cc,
                             edge_ids=edge_ids_cc,
                             edge_attr_labels=edge_attr_labels_cc,
                             graph_id=f"graph_ppi_{graph_idx}")
        # print(protein_graph)
        protein_graph_list.append(protein_graph)
        graph_idx += 1

    ####################################################################################################################
    # [2.] Statistic of the connected components graphs ================================================================
    ####################################################################################################################
    # min_node_nr = min(node_sizes)
    # max_node_nr = max(node_sizes)
    # print(f"Min nr. of nodes: {min_node_nr}, max nr. of nodes: {max_node_nr}")

    # counter_nr_nodes = Counter(node_sizes)
    # print(f"Unique number of nodes: {counter_nr_nodes.keys()} "
    #      f"and thow many times they appear in graphs: {counter_nr_nodes.values()}")

    # statistics_histogram(node_sizes, "number of nodes")     # Plot histogram of nr. of nodes distribution ~~~~~~~~~~~~

    ####################################################################################################################
    # [3.] Code time duration ==========================================================================================
    ####################################################################################################################
    datetime_2 = datetime.now()
    code_time_duration = datetime_2 - datetime_1

    print(f"Code time duration: {code_time_duration}")

    # # # Return # # #
    print(len(protein_graph_list))

    return protein_graph_list
