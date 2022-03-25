"""
    Graph classification of KIRC RANDOM dataset

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-03-25
"""

import os
import pickle
import random

from actionable.gnn_actions import gnn_init_train, gnn_predict, gnn_retrain


# [1.] Transformation Experiment ::: From PPI to Pytorch_Graph ---------------------------------------------------------
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))

# [2.] Train the GNN for the first time --------------------------------------------------------------------------------
# performance_values_dict = gnn_init_train(dataset)

# [3.] Tryout the predict function -------------------------------------------------------------------------------------
dataset_len = len(dataset)
graph_idx = random.randint(0, dataset_len)
input_graph = dataset[graph_idx]
gnn_predict(input_graph)


