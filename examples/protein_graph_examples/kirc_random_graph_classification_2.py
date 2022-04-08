"""
    Graph classification of KIRC RANDOM dataset

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-03-25
"""

import os
import pickle
import random

import numpy as np
import torch
from torch_geometric.data import Data

from actionable.gnn_actions import GNN_Actions
from actionable.gnn_explanations import explain_sample
from actionable.graph_actions import add_node, remove_node, remove_edge

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

# [1.] Transformation Experiment ::: From PPI to Pytorch_Graph ---------------------------------------------------------
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))

# [2.] Train the GNN for the first time --------------------------------------------------------------------------------
gnn_actions_obj = GNN_Actions()
performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)

# [3.] Tryout the predict function -------------------------------------------------------------------------------------
dataset_len = len(dataset)
graph_idx = random.randint(0, dataset_len)
input_graph = dataset[graph_idx]
print(f"Node that will be removed. {input_graph.node_labels[graph_idx]}")
print(input_graph.x.shape, input_graph.edge_index.shape)
input_graph_update = remove_node(input_graph, graph_idx, input_graph.node_labels[graph_idx])
# input_graph_update = remove_edge(input_graph, 1125, 502)
# edge_indexes = input_graph_update.edge_index.cpu().detach().numpy()
# print(edge_indexes.shape)
# edge_indexes = edge_indexes[:, edge_indexes[0] != 0]
# edge_indexes = edge_indexes[:, edge_indexes[1] != 0]
# if 0 in edge_indexes:
#     print("EDGE")
# edge_indexes = edge_indexes - 1
# input_graph_update_2 = Data(x=input_graph_update.x,
#                            edge_index=input_graph_update.edge_index,
#                            edge_attr=None,
#                            y=input_graph_update.y
#                            )
# print(input_graph_update_2.x.dtype)
predicted_class = gnn_actions_obj.gnn_predict(input_graph_update)
print(f"Predicted class: {predicted_class}")

# [4.] Explanation -----------------------------------------------------------------------------------------------------
explanation_method = 'saliency'     # Also possible: 'ig' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ground_truth_label = int(input_graph.y.cpu().detach().numpy()[0])
explanation_label = ground_truth_label  # Can also be the opposite - all possible combinations of 0 and 1 ~~~~~~~~~~~~~~

rel_pos = list(explain_sample(
        explanation_method,
        input_graph,
        explanation_label,
    ))
rel_pos = [str(round(edge_relevance, 2)) for edge_relevance in rel_pos]

print(rel_pos)
print(type(rel_pos[0]))

