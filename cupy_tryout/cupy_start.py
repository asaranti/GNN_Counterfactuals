"""
    CuPy test

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-07-14
"""

from datetime import datetime
import os
import pickle
import random

import numpy as np
import uuid

from actionable.gnn_actions import GNN_Actions
from actionable.graph_actions import add_node, remove_node, remove_edge

# [0.] -----------------------------------------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

# [1.] Transformation Experiment ::: From PPI to Pytorch_Graph ---------------------------------------------------------
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))
print(f"==================> Length of dataset: {len(dataset)}")

# [2.] Train the GNN for the first time --------------------------------------------------------------------------------
# gnn_actions_obj = GNN_Actions()
# performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)

# [3.] Do an action !!! ------------------------------------------------------------------------------------------------
dataset_len = len(dataset)
graph_idx = random.randint(0, dataset_len - 1)
input_graph = dataset[graph_idx]
print(input_graph)

nodes_nr = input_graph.x.shape[0]
node_features_size = input_graph.x.size(dim=1)
print(f"Nr. of nodes: {nodes_nr}")
node_idx = random.randint(0, nodes_nr - 1)
print(f"Node that will be removed. {input_graph.node_labels[node_idx]}")

node_features = np.random.randn(1, node_features_size).astype(np.float32)
node_id = str(uuid.uuid4())
label = "label_" + str(uuid.uuid4())

time_start_add_node = datetime.now().strftime("%H:%M:%S.%f")
input_graph_update = add_node(input_graph,
                              node_features,
                              label,
                              node_id)
time_end_add_node = datetime.now().strftime("%H:%M:%S.%f")

print(f"Start: {time_start_add_node}")
print(f"End: {time_end_add_node}")

print(input_graph_update)

