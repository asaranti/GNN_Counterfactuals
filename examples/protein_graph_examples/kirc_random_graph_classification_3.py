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

#os.environ["CUDA_VISIBLE_DEVICES"]=""
#device = torch.device("cpu")
#device = 'cpu'

# [1.] Transformation Experiment ::: From PPI to Pytorch_Graph ---------------------------------------------------------
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))
print(f"==================> Length of dataset: {len(dataset)}")
print(type(dataset[0].node_labels))

########################################################################################################################
########################################################################################################################
########################################################################################################################
# dataset_from_ui_folder = os.path.join("data", "input", "KIRC_RANDOM", "kirc_random_dict_from_ui")
# dataset_1 = pickle.load(open(os.path.join(dataset_from_ui_folder, 'kirc_pytorch_dic_multiple_graphs.pkl'), "rb"))
# print(dataset_1)

# dataset_2 = keep_only_first_graph_dataset(dataset)
# print(len(dataset_2))
# print(dataset_2)

# [2.] Train the GNN for the first time --------------------------------------------------------------------------------
gnn_actions_obj = GNN_Actions()
performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)

# [3.] Tryout the predict function -------------------------------------------------------------------------------------
dataset_len = len(dataset)
graph_idx = random.randint(0, dataset_len - 1)
input_graph = dataset[graph_idx]

# [3.1.] Delete the first node -----------------------------------------------------------------------------------------
nodes_nr = input_graph.x.shape[0]
print(f"Nr. of nodes: {nodes_nr}")
node_idx = random.randint(0, nodes_nr - 1)
print(f"Node that will be removed. {input_graph.node_labels[node_idx]}")
input_graph_update = remove_node(input_graph, node_idx)
predicted_class = gnn_actions_obj.gnn_predict(input_graph_update)
print(f"Predicted class: {predicted_class}")

########################################################################################################################
########################################################################################################################
########################################################################################################################

# [3.2.] Delete the second node ----------------------------------------------------------------------------------------
nodes_nr = input_graph_update.x.shape[0]
print(f"Nr. of nodes: {nodes_nr}")
node_idx = random.randint(0, nodes_nr - 1)
print(f"Node that will be removed. {input_graph_update.node_labels[node_idx]}")
input_graph_update_2 = remove_node(input_graph_update, node_idx)
predicted_class = gnn_actions_obj.gnn_predict(input_graph_update_2)
print(f"Predicted class: {predicted_class}")

print(input_graph_update_2)

# [4.] Re-train --------------------------------------------------------------------------------------------------------
input_graph_update_2_new_id = copy.deepcopy(input_graph_update_2)
input_graph_update_2_new_id.graph_id = f"graph_id_{graph_idx}_10"
zero_graph_new_id = copy.deepcopy(dataset[0])
zero_graph_new_id.graph_id = f"graph_id_0_5"
dataset.append(zero_graph_new_id)
dataset.append(input_graph_update_2_new_id)
# print(f"=================> Length of dataset: {len(dataset)}")
# output_dataset = keep_only_first_graph_dataset(dataset)
# print(f"=================> Length of dataset: {len(output_dataset)}")
test_set_metrics_dict = gnn_actions_obj.gnn_retrain(dataset)        # Re-train -----------------------------------------

print("After ReTrain:")
print(test_set_metrics_dict)

"""
# [5.] Explanation -----------------------------------------------------------------------------------------------------
# explanation_method = 'saliency'     # Also possible: 'ig' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ground_truth_label = int(input_graph.y.cpu().detach().numpy()[0])
# explanation_label = ground_truth_label  # Can also be the opposite - all possible combinations of 0 and 1 ~~~~~~~~~~~~

# rel_pos = list(explain_sample(
#        explanation_method,
#        input_graph,
#        explanation_label,
#    ))
# rel_pos = [str(round(edge_relevance, 2)) for edge_relevance in rel_pos]

# print(rel_pos)
# print(type(rel_pos[0]))
"""
