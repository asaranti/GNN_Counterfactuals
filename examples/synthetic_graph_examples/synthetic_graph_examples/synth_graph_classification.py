"""
    Graph classification of Synthetic dataset

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-04-28
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

from plots.graph_explanations_visualization import integrated_gradients_viz

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

########################################################################################################################
# [1.] Transformation Experiment ::: From PPI to Pytorch_Graph ---------------------------------------------------------
########################################################################################################################
dataset_pytorch_folder = os.path.join("data", "output", "Synthetic", "synthetic_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'synthetic_pytorch.pkl'), "rb"))

########################################################################################################################
# [2.] Train the GNN for the first time --------------------------------------------------------------------------------
########################################################################################################################
gnn_actions_obj = GNN_Actions()
performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)

print(performance_values_dict)

dataset_len = len(dataset)
graph_idx = random.randint(0, dataset_len - 1)  # Pick a random graph from the dataset ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
input_graph = dataset[graph_idx]

print(input_graph.edge_index)

"""
########################################################################################################################
# [3.] Get some patient info -------------------------------------------------------------------------------------------
########################################################################################################################
ground_truth_label = str(input_graph.y.cpu().detach().numpy()[0])

# Check if it is in the training or test dataset -----------------------------------------------------------------------
current_graph_id = input_graph.graph_id
b_is_in_train = gnn_actions_obj.is_in_training_set(current_graph_id)
which_dataset = "Test Data"
if b_is_in_train:
    which_dataset = "Training Data"

# Get its prediction label and prediction performance (or confidence for the prediction) -------------------------------
predicted_label, prediction_confidence = gnn_actions_obj.gnn_predict(input_graph)

print(which_dataset, ground_truth_label, predicted_label, prediction_confidence)
print(type(which_dataset), type(ground_truth_label), type(predicted_label), type(prediction_confidence))

"""
########################################################################################################################
# [5.] Explanation -----------------------------------------------------------------------------------------------------
# [5.1.] Compute the explanation values --------------------------------------------------------------------------------
########################################################################################################################
explanation_method = 'ig'     # Also possible: 'saliency' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ground_truth_label = int(input_graph.y.cpu().detach().numpy()[0])
explanation_label = ground_truth_label  # Can also be the opposite - all possible combinations of 0 and 1 ~~~~~~~~~~~~~~
predicted_class, prediction_confidence = gnn_actions_obj.gnn_predict(input_graph)

print(f"Graph idx: {graph_idx}, ground_truth_label: {ground_truth_label}, predicted class: {predicted_class}")

rel_pos = list(explain_sample(
        explanation_method,
        input_graph,
        explanation_label,
    ))
rel_pos = [str(round(edge_relevance, 2)) for edge_relevance in rel_pos]

print(rel_pos)
print(type(rel_pos[0]))

# [5.2.] Plot ----------------------------------------------------------------------------------------------------------
output_data_path = os.path.join(os.path.join("data", "output", "Synthetic", "plots", "explanations_plots",
                                             explanation_method
                                             ))

integrated_gradients_viz(input_graph, graph_idx, rel_pos, input_graph.node_labels, ground_truth_label,
                         int(predicted_class), explanation_label, output_data_path)

