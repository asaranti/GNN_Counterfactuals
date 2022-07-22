"""
    Format Transformation from the synthetic dataset to Pytorch
    Keep only a pre-specified number of graphs

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-07-22
"""

import copy
import os
import pickle
import random

from preprocessing_files.format_transformations.format_transformation_synth_to_pytorch import import_synthetic_data


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
# [2.] Select randomly a smaller amount of graphs ======================================================================
########################################################################################################################
graphs_nr = 50
synthetic_graph_50_list = random.choices(synthetic_graph_list, k=graphs_nr)

synthetic_graph_50_continuous_list = []
for graph_idx in range(len(synthetic_graph_50_list)):

    graph = copy.deepcopy(synthetic_graph_50_list[graph_idx])
    graph_id = f"graph_id_{graph_idx}_0"
    graph.graph_id = graph_id
    synthetic_graph_50_continuous_list.append(graph)

print(synthetic_graph_50_continuous_list)

########################################################################################################################
# [4.] Store the Pytorch Dataset =======================================================================================
########################################################################################################################
dataset_pytorch_folder = os.path.join("data", "output", "Synthetic", "synthetic_pytorch")
with open(os.path.join(dataset_pytorch_folder, 'synthetic_pytorch_50_graphs.pkl'), 'wb') as f:
    pickle.dump(synthetic_graph_50_continuous_list, f)

dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'synthetic_pytorch_50_graphs.pkl'), "rb"))
dataset_len = len(dataset)
print(dataset_len)

