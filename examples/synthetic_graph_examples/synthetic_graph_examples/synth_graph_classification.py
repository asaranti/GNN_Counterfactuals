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
print(type(dataset[0]))
print(dataset[0])

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

########################################################################################################################
# [3.] Let's do some actions -------------------------------------------------------------------------------------------
########################################################################################################################
"""
nodes_orig_nr = input_graph.x.shape[0]
print(f"Nr. of nodes: {nodes_orig_nr}")
edges_orig_nr = input_graph.edge_index.shape[1]
print(f"Nr. of edges: {edges_orig_nr}")

number_of_nodes_to_remain = 2

for loop_idx in range(nodes_orig_nr - number_of_nodes_to_remain):  # <<<<<<<<<<<<<<

    nodes_nr = input_graph.x.shape[0]

    edge_index_current = input_graph.edge_index.cpu().detach().numpy()
    edge_0 = edge_index_current[:, 0]
    print(f"Edge: {edge_0}")

    input_graph = remove_edge(input_graph, edge_0[0], edge_0[1])

    print(input_graph)
    edges_nr = input_graph.edge_index.size(dim=1)

    if edges_nr < 5:

        print("HERE")

        x_new = torch.tensor(np.array([[1.0], [4.0]]), dtype=torch.float32).to(device)  # dtype=torch.float64
        node_labels = ["1", "2"]
        node_ids = [1, 2]
        node_feature_labels = ["1", "2"]
        edge_ids_new = [504]
        edge_attr_labels = None
        input_graph = Data(x=x_new,
                           edge_index=torch.from_numpy(np.array([[0, 1]]).T),
                           y=input_graph.y,
                           node_labels=node_labels,
                           node_ids=node_ids,
                           node_feature_labels=node_feature_labels,
                           edge_ids=edge_ids_new,
                           edge_attr_labels=edge_attr_labels,
                           graph_id=input_graph.graph_id)

        predicted_class = gnn_actions_obj.gnn_retrain(dataset)
        print(f"Predicted class: {predicted_class}")

    elif edges_nr < 10:

        predicted_class = gnn_actions_obj.gnn_retrain(dataset)
        print(f"Predicted class: {predicted_class}")
        print("-------------------------------------------------------------------------------------------------------")

    else:
        predicted_class = gnn_actions_obj.gnn_predict(input_graph)
        print(f"Predicted class: {predicted_class}")
        print("-------------------------------------------------------------------------------------------------------")

    print("Resulting graph:")
    print(input_graph)
"""

########################################################################################################################
# [4.] Get some patient info -------------------------------------------------------------------------------------------
########################################################################################################################
ground_truth_label = str(input_graph.y.cpu().detach().numpy()[0])

# Check if it is in the training or test dataset -----------------------------------------------------------------------
for try_out in range(10):

    graph_idx = random.randint(0, dataset_len - 1)  # Pick a random graph from the dataset ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    input_graph = dataset[graph_idx]
    current_graph_id = input_graph.graph_id
    b_is_in_train = gnn_actions_obj.is_in_training_set(current_graph_id)
    print(f">>>>>>>>>> Is graph in training dataset? {b_is_in_train}")
    which_dataset = "Test Data"
    if b_is_in_train:
        which_dataset = "Training Data"

    # Get its prediction label and prediction performance (or confidence for the prediction) ---------------------------
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
"""
