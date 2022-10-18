"""
    Graph classification of KIRC RANDOM dataset
    Tryout feature addition and removal in nodes


    :author: Anna Saranti
    :copyright: © 2022 HCI-KDD (ex-AI) group
    :date: 2022-08-03
"""

import os
import pickle
import random

from actionable.gnn_actions import GNN_Actions
from actionable.graph_actions import add_node, remove_node

# [0.] -----------------------------------------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

# [1.] Transformation Experiment ::: From PPI to Pytorch_Graph ---------------------------------------------------------
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))

# [2.] Train the GNN for the first time --------------------------------------------------------------------------------
gnn_actions_obj = GNN_Actions()
performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)

# [3.] Delete one node -------------------------------------------------------------------------------------------------
dataset_len = len(dataset)
graph_idx = random.randint(0, dataset_len - 1)
input_graph = dataset[graph_idx]
print(input_graph)

nodes_orig_nr = input_graph.x.shape[0]
print(f"Nr. of nodes original: {nodes_orig_nr}")

node_idx = 0
output_graph = remove_node(input_graph, node_idx)
nodes_output_nr = output_graph.x.shape[0]
print(f"Nr. of nodes after node delete: {nodes_output_nr}")

dataset[graph_idx] = output_graph

# [4.] Retrain ---------------------------------------------------------------------------------------------------------
performance_values_dict = gnn_actions_obj.gnn_retrain(dataset)
print(performance_values_dict)
