"""
    Graph classification of KIRC RANDOM dataset #TUDataset

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-01-27
"""

import os
import random

import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import minmax_scale

from gnns.gnns_graph_classification.GCN_Graph_Classification import GCN
from preprocessing.format_transformations.format_transformation_random_kirc_to_pytorch import import_random_kirc_data
from sklearn.preprocessing import StandardScaler


########################################################################################################################
# [1.] Transformation Experiment ::: From PPI to Pytorch_Graph =========================================================
########################################################################################################################

dataset_folder = os.path.join("data", "KIRC_RANDOM", "kirc_random_orig")
pytorch_random_kirc_edges_file = "KIDNEY_RANDOM_PPI.txt"
pytorch_random_kirc_mRNA_attribute_file = "KIDNEY_RANDOM_mRNA_FEATURES.txt"
pytorch_random_kirc_methy_attribute_file = "KIDNEY_RANDOM_Methy_FEATURES.txt"
pytorch_random_kirc_target_file = "KIDNEY_RANDOM_TARGET.txt"

dataset = import_random_kirc_data(dataset_folder,
                                  pytorch_random_kirc_mRNA_attribute_file,
                                  pytorch_random_kirc_methy_attribute_file,
                                  pytorch_random_kirc_edges_file,
                                  pytorch_random_kirc_target_file)

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
    x_features_transformed = minmax_scale(x_features_array, feature_range=(-1, 1))
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
model = GCN(num_node_features=num_features, hidden_channels=200, num_classes=num_classes)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train():
    """
    Train ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    model.train()

    for data in train_loader:                               # Iterate in batches over the training dataset.

        out = model(data.x, data.edge_index, data.batch)    # Perform a single forward pass.
        loss = criterion(out, data.y)                       # Compute the loss.
        pred = out.argmax(dim=1)

        loss.backward()                                     # Derive gradients.
        optimizer.step()                                    # Update parameters based on gradients.
        optimizer.zero_grad()                               # Clear gradients.


def test(loader):
    """
    Test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    model.eval()

    correct = 0
    for data in loader:                                     # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)                            # Use the class with highest probability.

        correct += int((pred == data.y).sum())              # Check against ground-truth labels.
    return correct / len(loader.dataset)                    # Derive ratio of correct predictions.


# Training for some epochs +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
epochs_nr = 50
for epoch in range(1, epochs_nr + 1):

    print(f"Epoch: {epoch}")

    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)

    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    print("-------------------------------------------------------------------------")

