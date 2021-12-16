"""
    Format Transformation

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-11-22
"""

import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


########################################################################################################################
# [1.] Nodes ===========================================================================================================
########################################################################################################################
protein_dataset_folder = os.path.join("data", "Protein_Dataset")

human_tcga = open(os.path.join(protein_dataset_folder,
                  "Human__TCGA_ACC__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct"), "r")

node_attributes_orig = pd.read_csv(human_tcga, sep='\t')
print(node_attributes_orig.head(5))

# Not mandatory: remove "attribute_name" column before GNN training ----------------------------------------------------
node_attributes = node_attributes_orig.drop(['attrib_name'], axis=1)

human_tcga.close()

# # #
attrib_name_list = node_attributes_orig['attrib_name'].tolist()
row_indexes_list = list(range(0, len(attrib_name_list)))
attrib_name_row_indexes_dict = dict(zip(attrib_name_list, row_indexes_list))

########################################################################################################################
# [2.] Mapping for edges ===============================================================================================
########################################################################################################################
human_names = open(os.path.join(protein_dataset_folder, "human.name_2_string.csv"), "r")
human_names_proteins = {}
line_cnt = 0
for line in human_names:
    line_array = line.split('\t')

    if line_cnt >= 1:
        protein_name_2 = line_array[2].rstrip('\n')
        human_names_proteins[protein_name_2] = line_array[1]

    line_cnt += 1

########################################################################################################################
# [3.] Edges ===========================================================================================================
########################################################################################################################
edge_idx_left = []
edge_idx_right = []

edge_attr = []

protein_links = open(os.path.join(protein_dataset_folder, "9606.protein.links.v11.0.txt"), "r")
line_cnt = 0
for line in protein_links:

    if line_cnt >= 1:

        line_array = line.split('\t')
        line_edges_array = line.split(" ")

        # 3.1. Create the edge indexes =================================================================================
        protein_left_name = line_edges_array[0]
        protein_right_name = line_edges_array[1]

        # Protein left and protein right must have "human readable" names. If not the edge will not be added. ----------
        if protein_left_name in human_names_proteins.keys() and protein_right_name in human_names_proteins.keys():

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Using the Dataframe lasts longer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # protein_left_indexes_list = node_attributes_orig.index[node_attributes_orig['attrib_name'] ==
            #                                                       human_names_proteins[protein_left_name]].tolist()
            # protein_right_indexes_list = node_attributes_orig.index[node_attributes_orig['attrib_name'] ==
            #                                                        human_names_proteins[protein_right_name]].tolist()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if human_names_proteins[protein_left_name] in attrib_name_row_indexes_dict.keys() and \
                    human_names_proteins[protein_right_name] in attrib_name_row_indexes_dict.keys():

                protein_left_idx = attrib_name_row_indexes_dict[human_names_proteins[protein_left_name]]
                protein_right_idx = attrib_name_row_indexes_dict[human_names_proteins[protein_right_name]]

                # if len(protein_left_indexes_list) == 1 and len(protein_right_indexes_list) == 1:

                # protein_left_idx = protein_left_indexes_list[0]
                # protein_right_idx = protein_right_indexes_list[0]

                edge_idx_left.append(protein_left_idx)
                edge_idx_right.append(protein_right_idx)

        # 3.2. Create the edge attributes ==============================================================================
        edge_attr.append([float(line_edges_array[2].rstrip('\n'))])

    # if line_cnt >= 100:
    #    break

    line_cnt += 1

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
print(f"edge_idx: {[edge_idx_left, edge_idx_right]}")
print(f"edge_attr: {edge_attr}")

edge_idx = torch.tensor([edge_idx_left, edge_idx_right], dtype=torch.long)
edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float64)

########################################################################################################################
# [4.] Graph ===========================================================================================================
########################################################################################################################
protein_graph = Data(x=node_attributes, edge_index=edge_idx, edge_attr=edge_attr)
