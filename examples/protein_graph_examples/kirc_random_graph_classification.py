"""
    Graph classification of KIRC RANDOM dataset #TUDataset

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-01-27
"""

import os
import pickle
import random

import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import minmax_scale

from gnns.gnns_graph_classification.gnn_train_test_methods import train, test
from gnns.gnns_graph_classification.GCN_Graph_Classification import GCN
from preprocessing.format_transformations.format_transformation_random_kirc_to_pytorch import import_random_kirc_data
from sklearn.preprocessing import StandardScaler


########################################################################################################################
# [1.] Transformation Experiment ::: From PPI to Pytorch_Graph =========================================================
########################################################################################################################
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))


########################################################################################################################
# [2.] Data Preparation ================================================================================================
########################################################################################################################

# [2.1.] Input features preprocessing/normalization --------------------------------------------------------------------
for graph in dataset:

    x_features = graph.x
    x_features_array = x_features.cpu().detach().numpy()

    # scaler = StandardScaler()
    # scaler.fit(x_features_array)
    # x_features_transformed = scaler.transform(x_features_array)
    x_features_transformed = minmax_scale(x_features_array, feature_range=(0, 1))
    graph.x = torch.tensor(x_features_transformed)

# [2.2.] Split training/validation/test set ----------------------------------------------------------------------------
graph_0 = dataset[0]
num_features = graph_0.num_node_features
graphs_nr = len(dataset)
random.shuffle(dataset)

train_dataset_len = int(graphs_nr*3/4)
train_dataset = dataset[:train_dataset_len]
test_dataset = dataset[train_dataset_len:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

########################################################################################################################
# [3.] Graph Classification ============================================================================================
########################################################################################################################
num_classes = 2
model = GCN(num_node_features=num_features, hidden_channels=64, num_classes=num_classes)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training for some epochs ---------------------------------------------------------------------------------------------
epochs_nr = 20
for epoch in range(1, epochs_nr + 1):

    print(f"Epoch: {epoch}")

    train(model, train_loader, optimizer, criterion)
    train_acc = test(model, train_loader)
    test_acc = test(model, test_loader)

    print(f'Epoch: {epoch:03d}, Tra+in Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    print("-------------------------------------------------------------------------")

########################################################################################################################
# [4.] Store the GNN ===================================================================================================
########################################################################################################################
gnn_storage_folder = os.path.join("data", "output", "gnns")
gnn_model_file_path = os.path.join(gnn_storage_folder, "gcn_model.pth")
torch.save(model, gnn_model_file_path)

