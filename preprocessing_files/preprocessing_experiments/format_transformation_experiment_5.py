"""
    Format Transformation from the synthetic dataset to Pytorch
    From synthetic to Pytorch - keep only the first 50 graphs
    for testing purposes

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-07-22
"""

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

graphs_nr = 50
synthetic_graph_smaller_list = random.choices(synthetic_graph_list, k=graphs_nr)
print(synthetic_graph_smaller_list)
print(len(synthetic_graph_smaller_list))

########################################################################################################################
# [4.] Store the Pytorch Dataset =======================================================================================
########################################################################################################################
dataset_pytorch_folder = os.path.join("data", "output", "Synthetic", "synthetic_pytorch")
with open(os.path.join(dataset_pytorch_folder, 'synthetic_pytorch_50_graphs.pkl'), 'wb') as f:
    pickle.dump(synthetic_graph_list, f)
