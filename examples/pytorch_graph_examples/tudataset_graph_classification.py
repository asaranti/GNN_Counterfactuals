"""
    TUDataset Graph classification

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-07-06
"""

import os
import random

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from gnns.gnns_graph_classification.GCN_Graph_Classification import GCN
from gnns.gnns_graph_classification.gnn_train_test_methods import train_model, use_trained_model
from preprocessing_data.graph_features_normalization import graph_features_normalization

########################################################################################################################
# [0.] Import dataset ==================================================================================================
########################################################################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

dataset = TUDataset(root='data/TUDataset', name='MUTAG')

num_features = dataset.num_features
num_classes = dataset.num_classes

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}, {type(dataset)}')
print(f'Number of features: {num_features}')
print(f'Number of classes: {num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('=============================================================')

print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

normalized_graphs_dataset = graph_features_normalization(dataset)

tu_dataset = []
for graph in normalized_graphs_dataset:
    print(graph)
    graph.to(device)
    tu_dataset.append(graph)

########################################################################################################################
# [1.] GNN =============================================================================================================
########################################################################################################################
torch.manual_seed(12345)
random.shuffle(tu_dataset)

train_dataset = tu_dataset[:150]
test_dataset = tu_dataset[150:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

model = GCN(num_node_features=num_features, hidden_channels=20, num_classes=num_classes).to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss().to(device)

epochs_nr = 100
for epoch in range(1, epochs_nr + 1):
    train_model(model, train_loader, optimizer, criterion)

train_set_metrics_dict, train_outputs_predictions_dict = use_trained_model(model, train_loader)
test_set_metrics_dict, test_outputs_predictions_dict = use_trained_model(model, test_loader)
print(test_set_metrics_dict)
