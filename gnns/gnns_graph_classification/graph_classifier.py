"""
    Graph Classifier

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-10-05
"""

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from gnns.gnns_graph_classification.GCN import GCN
from plots.graph_visualization import graph_viz

########################################################################################################################
# [1.] Import data, split them to training and test datasets ===========================================================
########################################################################################################################
dataset = TUDataset(root='data/TUDataset', name='MUTAG')

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)

print(data.x.shape)
print(data.edge_index)
print(data.edge_attr.shape)
print(data.y)
print(data.pos)

print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

graph_viz(data)
print('=============================================================')

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

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

########################################################################################################################
# [2.] GCN =============================================================================================================
########################################################################################################################
bad_nr_features = 3
model = GCN(num_node_features=dataset.num_features, hidden_channels=64, num_classes=dataset.num_classes)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train():

    model.train()

    for data in train_loader:                               # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)    # Perform a single forward pass.
        loss = criterion(out, data.y)                       # Compute the loss.
        loss.backward()                                     # Derive gradients.
        optimizer.step()                                    # Update parameters based on gradients.
        optimizer.zero_grad()                               # Clear gradients.


def test(loader):

    model.eval()

    correct = 0
    for data in loader:                                     # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)                            # Use the class with highest probability.
        correct += int((pred == data.y).sum())              # Check against ground-truth labels.
    return correct / len(loader.dataset)                    # Derive ratio of correct predictions.


for epoch in range(1, 101):

    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)

    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


