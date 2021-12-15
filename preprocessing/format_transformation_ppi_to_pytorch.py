"""
    Format Transformation
    From PPI to Pytorch

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-11-22
"""

import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


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
    # [1.] Nodes =======================================================================================================
    ####################################################################################################################
    pytorch_to_ppi_attributes_file = open(os.path.join(input_dataset_folder, pytorch_to_ppi_attributes_file), "r")

    node_attributes_orig = pd.read_csv(pytorch_to_ppi_attributes_file, sep='\t')

    # Not mandatory: remove "attribute_name" column before GNN training ------------------------------------------------
    node_attributes_drop_attr_name_df = node_attributes_orig.drop(['attrib_name'], axis=1)
    print(node_attributes_drop_attr_name_df.head(5))
    node_feature_labels = node_attributes_drop_attr_name_df.columns.values
    node_attributes = torch.tensor(node_attributes_drop_attr_name_df.values)
    print(node_attributes_orig.shape, len(node_feature_labels))

    pytorch_to_ppi_attributes_file.close()

    # Attributes names dictionary correspondence with row --------------------------------------------------------------
    attrib_name_list = node_attributes_orig['attrib_name'].tolist()
    row_indexes_list = list(range(0, len(attrib_name_list)))
    attrib_name_row_indexes_dict = dict(zip(attrib_name_list, row_indexes_list))
    print(len(attrib_name_row_indexes_dict))

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

    print(len(human_names_proteins))

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
            else:
                if not protein_left_name_has_human_name:
                    print(f"Protein's name: {protein_left_name} not found in {pytorch_to_ppi_node_id_to_name_file}.")
                if not protein_right_name_has_human_name:
                    print(f"Protein's name : {protein_left_name} not found in {pytorch_to_ppi_node_id_to_name_file}.")

        line_cnt += 1

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # print(f"edge_idx: {[edge_idx_left, edge_idx_right]}")
    # print(f"edge_attr: {edge_attr}")

    edge_idx = torch.tensor([edge_idx_left, edge_idx_right], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float64)

    ####################################################################################################################
    # [4.] Graph =======================================================================================================
    ####################################################################################################################
    node_ids_list = [f"node_id_{x}" for x in range(0, len(attrib_name_list))]

    print(len(attrib_name_list), len(node_ids_list), len(node_feature_labels), len(edge_ids), len(edge_attr_labels))
    protein_graph = Data(x=node_attributes, edge_index=edge_idx, edge_attr=edge_attr, y=None, pos=None,
                         node_labels=np.array(attrib_name_list),
                         node_ids=np.array(node_ids_list),
                         node_feature_labels=node_feature_labels,
                         edge_ids=edge_ids,
                         edge_attr_labels=edge_attr_labels,
                         graph_id="graph_0"
                         )

    return protein_graph
