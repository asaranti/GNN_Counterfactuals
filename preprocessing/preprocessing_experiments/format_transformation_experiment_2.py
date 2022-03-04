"""
    Format Transformation Experiment Nr. 2

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-12-17
"""

import os

from preprocessing.format_transformations.format_transformation_pytorch_to_ui import transform_from_pytorch_to_ui
from preprocessing.format_transformations.format_transformation_ui_to_pytorch import transform_from_ui_to_pytorch
from examples.synthetic_graph_examples.ba_graphs_generator import ba_graphs_gen

########################################################################################################################
# [1.] BA graphs generation ============================================================================================
########################################################################################################################
graphs_nr = 100
nodes_nr = 10
edges_per_node_nr = 2    # Number of edges to attach from a new node to existing nodes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

node_features_nr = 5
edge_features_nr = 4

graphs_list = ba_graphs_gen(graphs_nr, nodes_nr, edges_per_node_nr, node_features_nr, edge_features_nr)

########################################################################################################################
# [2.] Pytorch to UI transformation ====================================================================================
########################################################################################################################
dataset_folder = os.path.join("data", "BA_Dataset")

graph_idx = 0
for graph_idx in range(graphs_nr):

    transform_from_pytorch_to_ui(graphs_list[graph_idx],
                                 dataset_folder,
                                 f"ba_nodes_ui_format_{graph_idx}.csv",
                                 f"ba_edges_ui_format_{graph_idx}.csv")

    graph_idx += 1

########################################################################################################################
# [3.] UI to Pytorch transformation ====================================================================================
########################################################################################################################
graph_idx = 0
graph_back = transform_from_ui_to_pytorch(dataset_folder,
                                          f"ba_nodes_ui_format_{graph_idx}.csv",
                                          f"ba_edges_ui_format_{graph_idx}.csv"
                                          )
print(graphs_list[0])
print(graph_back)

