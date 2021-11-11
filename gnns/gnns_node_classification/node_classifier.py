"""
    Node Classifier - Main code in:
    https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX?usp=sharing

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-11-11
"""

import torch

from gnns.gnns_node_classification.GCN_Node_Classification import GCN
from gnns.gnns_explainers.gnn_explainer import GNNExplainer


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

    graph_0 = dataset[0]  # Get the first graph object. ----------------------------------------------------------------

    print()
    print(graph_0)
    print('===========================================================================================================')

    # Gather some statistics about the graph. --------------------------------------------------------------------------
    print(f'Number of nodes: {graph_0.num_nodes}')
    print(f'Number of edges: {graph_0.num_edges}')
    print(f'Average node degree: {graph_0.num_edges / graph_0.num_nodes:.2f}')
    print(f'Number of training nodes: {graph_0.train_mask.sum()}')
    print(f'Training node label rate: {int(graph_0.train_mask.sum()) / graph_0.num_nodes:.2f}')
    print(f'Has isolated nodes: {graph_0.has_isolated_nodes()}')
    print(f'Has self-loops: {graph_0.has_self_loops()}')
    print(f'Is undirected: {graph_0.is_undirected()}')

    ####################################################################################################################
    # [2.] GNN Training ================================================================================================
    ####################################################################################################################
    model = GCN(num_node_features=dataset.num_features, hidden_channels=16, num_classes=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def train():

        model.train()
        optimizer.zero_grad()                       # Clear gradients.
        out = model(graph_0.x, graph_0.edge_index)        # Perform a single forward pass.
        loss = criterion(out[graph_0.train_mask],
                         graph_0.y[graph_0.train_mask])   # Compute the loss solely based on the training nodes.
        loss.backward()                             # Derive gradients.
        optimizer.step()                            # Update parameters based on gradients.

        return loss

    def test():

        model.eval()
        out = model(graph_0.x, graph_0.edge_index)
        pred = out.argmax(dim=1)                    # Use the class with highest probability.
        test_correct = pred[graph_0.test_mask] == graph_0.y[graph_0.test_mask]   # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(graph_0.test_mask.sum())  # Derive ratio of correct predictions.

        return test_acc

    for epoch in range(1, 101):
        loss = train()
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    test_acc = test()
    print(f'Test Accuracy: {test_acc:.4f}')

    ####################################################################################################################
    # [3.] xAI method ==================================================================================================
    ####################################################################################################################

    # [3.1.] Apply Explainable AI method -------------------------------------------------------------------------------
    feat_mask_types = ['individual_feature', 'scalar', 'feature']
    feat_mask_type = 'feature'
    explainer = GNNExplainer(model, log=True, return_type='log_prob',
                             allow_edge_mask=True,
                             feat_mask_type=feat_mask_type)

    node_idx = 2
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, graph_0.x, graph_0.edge_index)
    if feat_mask_type == 'scalar':
        ax, G = explainer.visualize_subgraph(2, graph_0.edge_index, edge_mask, y=graph_0.y,
                                             threshold=0.8,
                                             node_alpha=node_feat_mask)

    else:
        edge_y = torch.randint(low=0, high=30, size=(graph_0.edge_index.size(1),))
        ax, G = explainer.visualize_subgraph(2, graph_0.edge_index, edge_mask, y=graph_0.y,
                                             edge_y=edge_y, threshold=0.8)

    # [3.2.] Gather the relevances and return them ---------------------------------------------------------------------
    #        Relevances of nodes ---------------------------------------------------------------------------------------
    graph_nodes_relevances_data = list(G.nodes(data=True))
    graph_nodes_relevances_dict = {}
    for node_idx in range(graph_0.num_nodes):
        graph_nodes_relevances_dict[node_idx] = 0.0
    for graph_node_relevance in graph_nodes_relevances_data:
        graph_nodes_relevances_dict[graph_node_relevance[0]] = graph_node_relevance[1]['y']

    # Relevances of edges ----------------------------------------------------------------------------------------------
    graph_edges_relevances_data = list(G.edges(data=True))
    graph_edges_relevances_dict = {}
    edge_index_pairs = graph_0.edge_index.cpu().detach().numpy()
    edge_mask_array = edge_mask.cpu().detach().numpy()
    edge_idx = 0
    for row_1, row_2 in zip(edge_index_pairs[0, :], edge_index_pairs[1, :]):
        graph_edges_relevances_dict[(row_1, row_2)] = edge_mask_array[edge_idx]
        edge_idx += 1

    return {"graph_nodes_relevances": graph_nodes_relevances_dict,
            "graph_edges_relevances": graph_edges_relevances_dict}

