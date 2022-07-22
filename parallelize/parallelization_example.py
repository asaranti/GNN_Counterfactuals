"""
    Parallelization example

    :author: Anna Saranti
    :copyright: © 2022 HCI-KDD (ex-AI) group
    :date: 2022-07-22
"""

import copy
import multiprocessing
from multiprocessing import Pool
import os
import pickle
import random

from preprocessing_files.format_transformations.format_transformation_synth_to_pytorch import import_synthetic_data


def change_graph_id(graph):

    graph_id_comp_array = graph.graph_id.split("_")
    patient_id = int(graph_id_comp_array[2])

    graph.graph_id = 'graph_id_' + str(patient_id + 100) + "_0"

    return graph


########################################################################################################################
# MAIN =================================================================================================================
########################################################################################################################
# [1.] Transformation Experiment ::: From Synthetic to Pytorch_Graph ===================================================
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

# [2.] Select randomly a smaller amount of graphs ======================================================================
processes_nr = 20

graphs_nr = 50
synthetic_graph_50_list = synthetic_graph_list[:graphs_nr]      # random.choices(synthetic_graph_list, k=graphs_nr)
print("Original graph dataset:")
print(synthetic_graph_50_list)

print("Parallelize now: ")
with Pool(processes_nr) as p:
    print(p.map(change_graph_id, synthetic_graph_50_list))


