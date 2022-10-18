"""
    Graph classification of KIRC RANDOM dataset
    Tryout several actions of node addition + removal,
    edge addition+removal

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
from torch_geometric.loader import DataLoader

from actionable.gnn_actions import GNN_Actions
from actionable.graph_actions import add_node, remove_node, remove_edge

# [0.] =================================================================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

"""
########################################################################################################################
########################################################################################################################
########################################################################################################################
# edge_index = torch.tensor([[0, 1, 1, 2],
#                           [1, 0, 2, 1]], dtype=torch.long)
edge_index = torch.from_numpy(np.empty((2, 0))).to(device)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
edge_ids = torch.tensor([], dtype=torch.int32)
# edge_ids = []

input_graph = Data(x=x, edge_index=edge_index, edge_ids=edge_ids)
testing_graph_list = [input_graph]
testing_graph_loader = DataLoader(testing_graph_list, batch_size=1, shuffle=False)

for data in testing_graph_loader:
    print("No edges OK ?")

"""
########################################################################################################################
########################################################################################################################
########################################################################################################################
# [1.] Transformation Experiment ::: From PPI to Pytorch_Graph ---------------------------------------------------------
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))
print(f"==================> Length of dataset: {len(dataset)}")
print(type(dataset[0].node_labels))

# [2.] Train the GNN for the first time --------------------------------------------------------------------------------
gnn_actions_obj = GNN_Actions()
performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)

# [3.] Delete all but the last node ------------------------------------------------------------------------------------
dataset_len = len(dataset)
graph_idx = random.randint(0, dataset_len - 1)
input_graph = dataset[graph_idx]
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

        x_new = torch.tensor(np.array([[1.0, 2.0], [4.0, 5.0]]), dtype=torch.float32).to(device)  # dtype=torch.float64
        node_labels = ["1", "2"]
        node_ids = [1, 2]
        node_feature_labels = ["1", "2"]
        edge_ids_new = [504]
        edge_attr_labels = [1]
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

