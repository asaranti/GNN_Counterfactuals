"""
    Format Transformation Experiment Nr. 6
    Transform a part of the KIRC_RANDOM dataset to pytorch-compatible format

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-08-02
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from plots.graph_visualization import graph_viz

########################################################################################################################
# [0.] Transformation Experiment ::: From PPI to Pytorch_Graph =========================================================
########################################################################################################################

dataset_folder = os.path.join("data", "input", "KIRC_RANDOM", "kirc_random_orig")
pytorch_random_kirc_edges_file = "KIDNEY_RANDOM_PPI.txt"
pytorch_random_kirc_mRNA_attribute_file = "KIDNEY_RANDOM_mRNA_FEATURES.txt"
pytorch_random_kirc_methy_attribute_file = "KIDNEY_RANDOM_Methy_FEATURES.txt"
pytorch_random_kirc_target_file = "KIDNEY_RANDOM_TARGET.txt"

########################################################################################################################
# [1.] mRNA and Methy attributes =======================================================================================
########################################################################################################################
node_feature_labels = ["mRNA", "methy"]

pytorch_random_kirc_mRNA_attribute_file = os.path.join(dataset_folder,
                                                       pytorch_random_kirc_mRNA_attribute_file)
node_attribute_mRNA_orig = pd.read_csv(pytorch_random_kirc_mRNA_attribute_file, sep=' ')
mRNA_nan_values = node_attribute_mRNA_orig.isnull().sum().sum()
assert mRNA_nan_values == 0, f"The number of NaN values in the features must be 0, " \
                                 f"instead it is: {mRNA_nan_values}"

pytorch_random_kirc_methy_attribute_file = os.path.join(dataset_folder,
                                                        pytorch_random_kirc_methy_attribute_file)
node_attribute_methy_orig = pd.read_csv(pytorch_random_kirc_methy_attribute_file, sep=' ')

# Check and deal with NaNs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
methy_nan_values = node_attribute_methy_orig.isnull().sum().sum()
# assert methy_nan_values == 0, f"The number of NaN values in the features must be 0, " \
#                              f"instead it is: {methy_nan_values}"

# node_attribute_methy_orig = node_attribute_methy_orig.fillna(node_attribute_methy_orig.mean())

column_has_nan = node_attribute_methy_orig.columns[node_attribute_methy_orig.isna().any()].tolist()
columns_with_nan = node_attribute_methy_orig[column_has_nan]
node_attribute_methy_processed = node_attribute_methy_orig.drop(columns_with_nan, axis=1)
node_attribute_mRNA_processed = node_attribute_mRNA_orig.drop(columns_with_nan, axis=1)

# Check if the node names in the two pandas dataframes are equal ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
node_names_mRNA_attribute_list = list(node_attribute_mRNA_processed.columns)
node_names_methy_attribute_list = list(node_attribute_methy_processed.columns)
node_names_same = node_names_mRNA_attribute_list == node_names_methy_attribute_list
assert node_names_same, f"Node names must be equal for both node attribute files: {node_names_same}"

# Check if the graph ids in the two pandas dataframes are equal ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
graph_ids_mRNA_list = node_attribute_mRNA_processed.index.tolist()
graph_ids_methy_list = node_attribute_mRNA_processed.index.tolist()

graph_ids_same = graph_ids_mRNA_list == graph_ids_methy_list
assert graph_ids_same, f"Graph ids must be equal for both node attribute files: {graph_ids_same}"

########################################################################################################################
# [2.] Nodes ===========================================================================================================
########################################################################################################################
# Only selected nodes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
allowed_nodes_names = ['MGAT3', 'MGAT4B', 'MGAT5', 'MGAT5B']
nr_of_nodes = len(allowed_nodes_names)
allowed_nodes_indexes = []

for allowed_node in allowed_nodes_names:

    allowed_node_index = node_names_mRNA_attribute_list.index(allowed_node)
    allowed_nodes_indexes.append(allowed_node_index)

print(allowed_nodes_indexes)

# Create the nodes attributes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nr_of_graphs = len(node_attribute_mRNA_processed.index)
node_attributes_list = []
for row_nr in range(nr_of_graphs):
    feature_mRNA = node_attribute_mRNA_processed.iloc[row_nr].values
    feature_methy = node_attribute_methy_processed.iloc[row_nr].values

    node_attributes = np.vstack((feature_mRNA.T, feature_methy.T)).T
    nodes_attributes_allowed = node_attributes[allowed_nodes_indexes, :]
    node_attributes_list.append(torch.tensor(nodes_attributes_allowed, dtype=torch.float32))

########################################################################################################################
# [3.] Edges ===========================================================================================================
########################################################################################################################
edges_list_original = [['5654629', 'MGAT3', 'MGAT4B', 983],
                       ['5654759 ', 'MGAT3', 'MGAT5', 892],
                       ['5654764', 'MGAT3', 'MGAT5B', 658],
                       ['5300888', 'MGAT4B', 'MGAT3', 983],
                       ['5300935', 'MGAT4B', 'MGAT5', 983],
                       ['5300937', 'MGAT4B', 'MGAT5B', 974],
                       ['9260860', 'MGAT5', 'MGAT4B', 983],
                       ['9260923', 'MGAT5', 'MGAT3', 892],
                       ['9261075', 'MGAT5', 'MGAT5B', 811],
                       ['9444587', 'MGAT5B', 'MGAT3', 658],
                       ['9444670', 'MGAT5B', 'MGAT5', 811],
                       ['9444805', 'MGAT5B', 'MGAT4B', 974]]

edges_left_indexes = []
edges_right_indexes = []
edge_attr = []
edge_ids = []

for edge_line in edges_list_original:

    edge_id = edge_line[0]
    node_left_name = edge_line[1]
    node_right_name = edge_line[2]
    combined_score = edge_line[3]

    if combined_score > 700:

        edge_attr.append(combined_score)

        node_left_idx = allowed_nodes_names.index(node_left_name)
        node_right_idx = allowed_nodes_names.index(node_right_name)

        edges_left_indexes.append(node_left_idx)
        edges_right_indexes.append(node_right_idx)

        edge_ids.append(edge_id)

edge_idx = torch.tensor([edges_left_indexes, edges_right_indexes], dtype=torch.long)
edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float64)

########################################################################################################################
# [4.] Label of each graph =============================================================================================
########################################################################################################################
pytorch_random_kirc_label_file = os.path.join(dataset_folder, pytorch_random_kirc_target_file)
random_kirc_target_orig = pd.read_csv(pytorch_random_kirc_label_file, sep=' ')

node_ids_list = [f"node_id_{x}" for x in range(0, nr_of_nodes)]

graph_all = []
for row_nr in range(nr_of_graphs):

    print(f"Graph Nr.: {row_nr}")

    graph_id = graph_ids_mRNA_list[row_nr]
    label = random_kirc_target_orig[graph_id].values[0]

    graph_orig = Data(
        x=node_attributes_list[row_nr],
        edge_index=edge_idx,
        edge_attr=None,  # edge_attr,
        y=torch.tensor([label]),
        pos=None,
        node_labels=np.array(allowed_nodes_names),
        node_ids=np.array(node_ids_list),
        node_feature_labels=node_feature_labels,
        edge_ids=edge_ids,
        edge_attr_labels=["combined_score"],
        graph_id=f"graph_id_{row_nr}_0"
    )

    """
    print(graph_orig.x)
    print(graph_orig.edge_index)
    print(graph_orig.y)
    print(graph_orig.node_labels)
    print(graph_orig.node_ids)
    print(graph_orig.node_feature_labels)
    print(graph_orig.edge_ids)
    print(graph_orig.edge_attr_labels)
    print("----------------------------------------------------------------------------------")
    """

    # print(graph_orig.edge_index)
    # graph_viz(graph_orig, row_nr)

    graph_all.append(graph_orig)

########################################################################################################################
# [5.] Store the Pytorch Dataset =======================================================================================
########################################################################################################################
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
with open(os.path.join(dataset_pytorch_folder, 'kirc_subnet_pytorch.pkl'), 'wb') as f:
    pickle.dump(graph_all, f)


