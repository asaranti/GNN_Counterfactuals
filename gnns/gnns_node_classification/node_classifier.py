"""
    Node Classifier - Main code in:
    https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-11-11
"""

import torch

from gnns.gnns_node_classification.GCN_Node_Classification import GCN
from torch_geometric.nn import GNNExplainer


def node_classification(dataset):
    """
    Perform node classification on a dataset
    """

    ####################################################################################################################
    # [1.] Dataset =====================================================================================================
    ####################################################################################################################

    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object. -------------------------------------------------------------------

    print()
    print(data)
    print('===========================================================================================================')

    # Gather some statistics about the graph. --------------------------------------------------------------------------
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    ####################################################################################################################
    # [2.] GNN Training ================================================================================================
    ####################################################################################################################
    model = GCN(num_node_features=dataset.num_features, hidden_channels=16, num_classes=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()
        optimizer.zero_grad()                       # Clear gradients.
        out = model(data.x, data.edge_index)        # Perform a single forward pass.
        loss = criterion(out[data.train_mask],
                         data.y[data.train_mask])   # Compute the loss solely based on the training nodes.
        loss.backward()                             # Derive gradients.
        optimizer.step()                            # Update parameters based on gradients.
        return loss

    def test():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)                    # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask]   # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
        return test_acc

    for epoch in range(1, 101):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    test_acc = test()
    print(f'Test Accuracy: {test_acc:.4f}')

    ####################################################################################################################
    # [3.] xAI method ==================================================================================================
    ####################################################################################################################
