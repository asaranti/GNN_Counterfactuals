"""
    Graph classification of KIRC RANDOM dataset
    Tryout feature addition and removal in nodes

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-07-21
"""

import os
import pickle
import random

import numpy as np

from actionable.gnn_actions import GNN_Actions
from actionable.graph_actions import add_feature_all_nodes, remove_feature_all_nodes, add_feature_all_edges,\
    remove_feature_all_edges

# [0.] -----------------------------------------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

# [1.] Transformation Experiment ::: From PPI to Pytorch_Graph ---------------------------------------------------------
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))

# [2.] Train the GNN for the first time --------------------------------------------------------------------------------
gnn_actions_obj = GNN_Actions()
performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)

# [3.] Remove node feature by index ------------------------------------------------------------------------------------
dataset_len = len(dataset)
graph_idx = random.randint(0, dataset_len - 1)
input_graph = dataset[graph_idx]

print(f"Graph before node feature removal: {input_graph}")
print(input_graph.edge_attr_labels)

x_numpy = input_graph.x.cpu().detach().numpy()
print(x_numpy)
print(input_graph.node_feature_labels)
print("----------------------------------------------------------------")

input_graph = remove_feature_all_nodes(input_graph, 0)
input_graph = remove_feature_all_nodes(input_graph, 0)

# [4.] Add node feature by index ---------------------------------------------------------------------------------------
new_feature_values = np.random.rand(x_numpy.shape[0], 1).astype('float32')
input_graph = add_feature_all_nodes(input_graph, new_feature_values, "new_node_feature_1")
new_feature_values = np.random.rand(x_numpy.shape[0], 1).astype('float32')
input_graph = add_feature_all_nodes(input_graph, new_feature_values, "new_node_feature_2")

print(f"Graph after node feature removal: {input_graph}")
x_numpy = input_graph.x.cpu().detach().numpy()
print(x_numpy)
print(input_graph.node_feature_labels)
print("----------------------------------------------------------------")

# [5.] Try to do a predict again ---------------------------------------------------------------------------------------
prediction_label_of_testing, prediction_confidence_of_testing = gnn_actions_obj.gnn_predict(input_graph)
print(prediction_label_of_testing, prediction_confidence_of_testing)

"""
# [6.] Remove edge feature by index ------------------------------------------------------------------------------------

# [7.] Add edge feature by index ---------------------------------------------------------------------------------------
print(f"Graph before edge feature add: {input_graph}")
edge_index_numpy = input_graph.x.cpu().detach().numpy()
edges_nr = edge_index_numpy.shape[1]

new_feature_values = np.random.rand(edges_nr, 1).astype('float32')
input_graph = add_feature_all_edges(input_graph, new_feature_values, "new_edge_feature_1")

print(f"Graph after edge feature add: {input_graph}")
"""
