"""
    The GNN train() and test() functionality

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-02-18
"""

import torch
from torch_geometric.loader.dataloader import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from gnns.gnns_graph_classification.GCN_Graph_Classification import GCN


def train(model: GCN, train_loader: DataLoader, optimizer, criterion):
    """
    Training of the model

    :param model: GNN model
    :param train_loader: Loader of training set
    :param optimizer: Model optimizer
    :param criterion: Loss criterion for the GNN training
    """

    model.train()

    for data in train_loader:                               # Iterate in batches over the training dataset.

        out = model(data.x, data.edge_index, data.batch)    # Perform a single forward pass.
        loss = criterion(out, data.y)                       # Compute the loss.
        pred = out.argmax(dim=1)

        loss.backward()                                     # Derive gradients.
        optimizer.step()                                    # Update parameters based on gradients.
        optimizer.zero_grad()                               # Clear gradients.


def test(model: GCN, test_loader: DataLoader):
    """
    Test the model - pass all data in the test_loader to the model
    to check its performance

    :param model: GNN model
    :param test_loader: Loader of test set
    """

    model.eval()

    correct = 0
    for data in test_loader:                                     # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)                            # Use the class with highest probability.

        correct += int((pred == data.y).sum())              # Check against ground-truth labels.
    return correct / len(test_loader.dataset)                    # Derive ratio of correct predictions.