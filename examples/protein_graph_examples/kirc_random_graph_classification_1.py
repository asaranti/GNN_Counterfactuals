"""
    Graph classification of KIRC RANDOM dataset

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-01-27
"""

from datetime import datetime
import os
import pickle
import random
import shutil

import networkx as nx
import numpy as np
from operator import itemgetter
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from actionable.gnn_explanations import explain_sample
from gnns.gnns_graph_classification.gnn_train_test_methods import train_model, use_trained_model
# from gnns.gnns_graph_classification.GCN_Graph_Classification import GCN
from gnns.gnns_graph_classification.GIN_Graph_Classification import GIN
from plots.graph_explanations_visualization import integrated_gradients_viz
from preprocessing_data.graph_features_normalization import graph_features_normalization

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

########################################################################################################################
# [1.] Transformation Experiment ::: From PPI to Pytorch_Graph =========================================================
########################################################################################################################
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
input_dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))

########################################################################################################################
# [2.] Data Preparation ================================================================================================
########################################################################################################################

# [2.1.] Input features preprocessing_files/normalization --------------------------------------------------------------
# normalized_graphs_dataset = graph_features_normalization(input_dataset)
normalized_graphs_dataset = []
for graph in input_dataset:
    graph.to(device)
    normalized_graphs_dataset.append(graph)

# [2.2.] Split training/validation/test set ----------------------------------------------------------------------------
graph_0 = normalized_graphs_dataset[0]
num_features = graph_0.num_node_features
graphs_nr = len(normalized_graphs_dataset)

# [2.3.] Shuffle the dataset and keep the list indexes -----------------------------------------------------------------
x = list(enumerate(normalized_graphs_dataset))
random.shuffle(x)
random_indices, graphs_list = zip(*x)
dataset_random_shuffling = list(itemgetter(*random_indices)(normalized_graphs_dataset))

# [2.4.] Split to training and test set --------------------------------------------------------------------------------
train_dataset_len = int(graphs_nr*3/4)
train_dataset = dataset_random_shuffling[:train_dataset_len]
test_dataset = dataset_random_shuffling[train_dataset_len:]

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
model = GIN(num_node_features=num_features, dim_h=100, num_classes=num_classes).to(device)
print(model)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = torch.nn.CrossEntropyLoss().to(device)

# Training for some epochs ---------------------------------------------------------------------------------------------
date_time_obj = datetime.now()
time_stamp_srt = date_time_obj.strftime("%d-%b-%Y %H:%M:%S")
print(f'Training time start: {time_stamp_srt}')


def train(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.01,
                                 weight_decay=0.01)
    epochs = 100

    model.train()
    for epoch in range(epochs + 1):

        print(f"Epoch: {epoch}")

        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0

        # Train on batches
        for data in loader:
            optimizer.zero_grad()
            _, out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss / len(loader)
            acc += accuracy(out.argmax(dim=1), data.y) / len(loader)
            loss.backward()
            optimizer.step()

            # Validation
            val_loss, val_acc = test(model, test_loader)

    # Print metrics every 10 epochs
    if (epoch % 10 == 0):
        print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
              f'| Train Acc: {acc * 100:>5.2f}% '
              f'| Val Loss: {val_loss:.2f} '
              f'| Val Acc: {val_acc * 100:.2f}%')

    test_loss, test_acc = test(model, test_loader)
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%')

    return model


def test(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    for data in loader:
        _, out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y) / len(loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

    return loss, acc


def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


gin = train(model, train_loader)

date_time_obj = datetime.now()
time_stamp_srt = date_time_obj.strftime("%d-%b-%Y %H:%M:%S")
print(f'Training time end: {time_stamp_srt}')
