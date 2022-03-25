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

from actionable.gnn_actions import gnn_init_train, gnn_predict, gnn_retrain
from actionable.gnn_explanations import explain_sample


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
