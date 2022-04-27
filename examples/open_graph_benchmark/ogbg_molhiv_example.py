"""
    Ogbg_Molhiv example

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-04-26
"""

from operator import itemgetter
import os
import random
import sys

import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.preprocessing import minmax_scale
import torch
from torch_geometric.data import DataLoader

from gnns.gnns_graph_classification.GCN_Graph_Classification import GCN
from gnns.gnns_graph_classification.gnn_train_test_methods import train_model, use_trained_model
from plots.graph_visualization import graph_viz_general

########################################################################################################################
# [0.] Import the dataset ==============================================================================================
########################################################################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

dataset_name = "ogbg-molhiv"
dataset = PygGraphPropPredDataset(name=dataset_name)

########################################################################################################################
# [1.] Analyze, visualize graph(s) of the dataset ======================================================================
########################################################################################################################
dataset_len = len(dataset)
print(f"Size of whole dataset: {dataset_len}")

class_0_cnt = 0
class_1_cnt = 0

x_min_all = sys.float_info.max
x_max_all = sys.float_info.min

edge_attr_min_all = sys.float_info.max
edge_attr_max_all = sys.float_info.min

for graph_idx in range(dataset_len):

    graph = dataset[graph_idx]

    # [1.1.] Classes (im)-balance --------------------------------------------------------------------------------------
    graph_y = graph.y.cpu().detach().numpy()
    y_class = graph_y[0][0]
    if y_class == 0:
        class_0_cnt += 1
    else:
        class_1_cnt += 1

    # [1.2.] Gather the [min] and [max] of node attributes for normalization -------------------------------------------
    graph_x = graph.x.cpu().detach().numpy()
    if x_max_all < np.amax(graph_x):
        x_max_all = np.amax(graph_x)
    if x_min_all > np.amin(graph_x):
        x_min_all = np.amin(graph_x)

    # [1.3.] Gather the [min] and [max] of edge attributes for normalization -------------------------------------------
    graph_edge_attr = graph.edge_attr.cpu().detach().numpy()
    if edge_attr_max_all < np.amax(graph_edge_attr):
        edge_attr_max_all = np.amax(graph_edge_attr)
    if edge_attr_min_all > np.amin(graph_edge_attr):
        edge_attr_min_all = np.amin(graph_edge_attr)

    # [1.4.] Visualization ---------------------------------------------------------------------------------------------
    # graph_viz_general(graph)

print(f"Class 0: {class_0_cnt}, Class 1: {class_1_cnt}")
print(f"Global \"x\" min: {x_min_all} and max: {x_max_all}")
print(f"Global \"edge_attr\" min: {edge_attr_min_all} and max: {edge_attr_max_all}")

########################################################################################################################
# [2.] Normalize =======================================================================================================
########################################################################################################################
normalized_dataset = []
for graph_idx in range(dataset_len):

    graph = dataset[graph_idx]

    graph_y = graph.y.cpu().detach().numpy()
    y_class = int(graph_y[0][0])
    graph.y = torch.tensor(np.array([y_class]))

    # [2.1.] Normalize the node attributes -----------------------------------------------------------------------------
    graph_x = graph.x.cpu().detach().numpy()
    x_features_transformed = minmax_scale(graph_x, feature_range=(0, 1))    # float32
    graph.x = torch.tensor(x_features_transformed).to(dtype=torch.float32)

    # [2.2.] Normalize the edge attributes -----------------------------------------------------------------------------
    # graph_edge_attr = graph.edge_attr.cpu().detach().numpy()
    # edge_attr_features_transformed = minmax_scale(graph_edge_attr, feature_range=(0, 1))
    graph.edge_attr = None  # torch.tensor(edge_attr_features_transformed)

    graph.to(device)
    normalized_dataset.append(graph)

########################################################################################################################
# [3.] Training ========================================================================================================
########################################################################################################################
graphs_nr = len(normalized_dataset)
x = list(enumerate(normalized_dataset))
random.shuffle(x)
random_indices, graphs_list = zip(*x)
dataset_random_shuffling = list(itemgetter(*random_indices)(normalized_dataset))

# [3.1.] Split to training and test set --------------------------------------------------------------------------------
proportion_of_training_set = 3/4
train_dataset_len = int(graphs_nr * proportion_of_training_set)
train_dataset = dataset_random_shuffling[:train_dataset_len]
test_dataset = dataset_random_shuffling[train_dataset_len:]
train_dataset_shuffled_indexes = random_indices[:train_dataset_len]
test_dataset_shuffled_indexes = random_indices[train_dataset_len:]

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# [3.2.] Classification ------------------------------------------------------------------------------------------------
num_classes = 2
graph_0 = normalized_dataset[0]
print(graph_0.x.dtype, graph_0.edge_index.dtype, graph_0.y.dtype, graph_0.y.shape)
num_features = graph_0.num_node_features

model = GCN(num_node_features=num_features, hidden_channels=20, num_classes=num_classes).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss().to(device)

epochs_nr = 20
for epoch in range(1, epochs_nr + 1):
    train_model(model, train_loader, optimizer, criterion)

train_set_metrics_dict, train_outputs_predictions_dict = use_trained_model(model, train_loader)
test_set_metrics_dict, test_outputs_predictions_dict = use_trained_model(model, test_loader)
print(test_set_metrics_dict)

