"""
    Graph classification of KIRC RANDOM dataset

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-03-25
"""

import copy
import os
import pickle
import random

import numpy as np
import torch
from torch_geometric.data import Data

from actionable.gnn_actions import GNN_Actions
from actionable.gnn_explanations import explain_sample
from actionable.graph_actions import add_node, remove_node, remove_edge

from utils.dataset_utilities import keep_only_first_graph_dataset, keep_only_last_graph_dataset

# [0.] =================================================================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

# os.environ["CUDA_VISIBLE_DEVICES"]=""
# device = torch.device("cpu")
# device = 'cpu'

# [1.] Transformation Experiment ::: From PPI to Pytorch_Graph ---------------------------------------------------------
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))
print(f"==================> Length of dataset: {len(dataset)}")
print(type(dataset[0].node_labels))

# [2.] Train the GNN for the first time --------------------------------------------------------------------------------
gnn_actions_obj = GNN_Actions()
performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)

"""
# [3.] Tryout the predict function -------------------------------------------------------------------------------------
dataset_len = len(dataset)
graph_idx = random.randint(0, dataset_len - 1)
input_graph = dataset[graph_idx]

# [4.] Explanation -----------------------------------------------------------------------------------------------------
explanation_method = 'gnnexplainer'     # Also possible: 'ig' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ground_truth_label = int(input_graph.y.cpu().detach().numpy()[0])
explanation_label = ground_truth_label  # Can also be the opposite - all possible combinations of 0 and 1 ~~~~~~~~~~~~

# GNNECPLAINER
node_mask = explain_sample(explanation_method , input_graph, explanation_label)
len(node_mask)

# CAPTUM 
# rel_pos = list(explain_sample(
#        explanation_method,
#        input_graph,
#        explanation_label,
#    ))
# rel_pos = [str(round(edge_relevance, 2)) for edge_relevance in rel_pos]

# print(rel_pos)
# print(type(rel_pos[0]))
"""
