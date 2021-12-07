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


class GNNNodeClassifierExplainer:
    """
    GNN Node classifier and explainer
    """

    model = None    # The GNN model that will be trained and explained -------------------------------------------------
    graph_idx = 0   # The selected graph on which the node will be selected and explained ------------------------------

    def node_classification(self, dataset, graph_idx: int):
        """
        Perform node classification on a dataset

        :param dataset: Dataset with graphs
        :param graph_idx: Graph index
        """

        self.graph_idx = graph_idx

        ################################################################################################################
        # [1.] Dataset =================================================================================================
        ################################################################################################################

        print()
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')

        selected_graph = dataset[self.graph_idx]  # Get the selected graph object. -------------------------------------

        print()
        print(selected_graph)
        print('=======================================================================================================')

        # Gather some statistics about the graph. ----------------------------------------------------------------------
        print(f'Number of nodes: {selected_graph.num_nodes}')
        print(f'Number of edges: {selected_graph.num_edges}')
        print(f'Average node degree: {selected_graph.num_edges / selected_graph.num_nodes:.2f}')
        print(f'Number of training nodes: {selected_graph.train_mask.sum()}')
        print(f'Training node label rate: {int(selected_graph.train_mask.sum()) / selected_graph.num_nodes:.2f}')
        print(f'Has isolated nodes: {selected_graph.has_isolated_nodes()}')
        print(f'Has self-loops: {selected_graph.has_self_loops()}')
        print(f'Is undirected: {selected_graph.is_undirected()}')

        ################################################################################################################
        # [2.] GNN Training ============================================================================================
        ################################################################################################################
        self.model = GCN(num_node_features=dataset.num_features, hidden_channels=16, num_classes=dataset.num_classes)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        def train():

            self.model.train()
            optimizer.zero_grad()                               # Clear gradients.
            out = self.model(selected_graph.x, selected_graph.edge_index)     # Perform a single forward pass.
            loss = criterion(out[selected_graph.train_mask],
                             selected_graph.y[selected_graph.train_mask])     # Compute the loss solely based on the training nodes.
            loss.backward()                                     # Derive gradients.
            optimizer.step()                                    # Update parameters based on gradients.

            return loss

        def test():

            self.model.eval()
            out = self.model(selected_graph.x, selected_graph.edge_index)
            pred = out.argmax(dim=1)                            # Use the class with highest probability.
            test_correct = pred[selected_graph.test_mask] == selected_graph.y[selected_graph.test_mask]  # Check against ground-truth labels.
            test_acc = int(test_correct.sum()) / int(selected_graph.test_mask.sum())  # Derive ratio of correct predictions.

            return test_acc

        for epoch in range(1, 101):
            loss = train()
            # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        test_acc = test()
        print(f'Test Accuracy: {test_acc:.4f}')

    def node_explanation(self, dataset, node_idx: int):
        """
        Node explanation

        :param dataset: Dataset with graphs
        :param node_idx: Node that is requested to be explained
        """

        selected_graph = dataset[self.graph_idx]  # Get the selected graph object. -------------------------------------

        ################################################################################################################
        # xAI method ===================================================================================================
        ################################################################################################################

        # [1.] Apply Explainable AI method -----------------------------------------------------------------------------
        feat_mask_types = ['individual_feature', 'scalar', 'feature']
        feat_mask_type = 'feature'
        explainer = GNNExplainer(self.model, log=True, return_type='log_prob',
                                 allow_edge_mask=True,
                                 feat_mask_type=feat_mask_type)

        node_feat_mask, edge_mask = explainer.explain_node(node_idx, selected_graph.x, selected_graph.edge_index)
        if feat_mask_type == 'scalar':
            ax, G = explainer.visualize_subgraph(node_idx, selected_graph.edge_index, edge_mask, y=selected_graph.y,
                                                 threshold=0.8,
                                                 node_alpha=node_feat_mask)

        else:
            edge_y = torch.randint(low=0, high=30, size=(selected_graph.edge_index.size(1),))
            ax, G = explainer.visualize_subgraph(node_idx, selected_graph.edge_index, edge_mask, y=selected_graph.y,
                                                 edge_y=edge_y, threshold=0.8)

        # [2.] Gather the relevances and return them -------------------------------------------------------------------
        #      Relevances of nodes -------------------------------------------------------------------------------------
        graph_nodes_relevances_data = list(G.nodes(data=True))
        graph_nodes_relevances_dict = {}
        for node_idx in range(selected_graph.num_nodes):
            graph_nodes_relevances_dict[node_idx] = 0.0
        for graph_node_relevance in graph_nodes_relevances_data:
            graph_nodes_relevances_dict[graph_node_relevance[0]] = graph_node_relevance[1]['y']

        # [3.] Relevances of edges -------------------------------------------------------------------------------------
        graph_edges_relevances_data = list(G.edges(data=True))
        graph_edges_relevances_dict = {}
        edge_index_pairs = selected_graph.edge_index.cpu().detach().numpy()
        edge_mask_array = edge_mask.cpu().detach().numpy()
        edge_idx = 0
        for row_1, row_2 in zip(edge_index_pairs[0, :], edge_index_pairs[1, :]):
            graph_edges_relevances_dict[(row_1, row_2)] = edge_mask_array[edge_idx]
            edge_idx += 1

        return {"graph_nodes_relevances": graph_nodes_relevances_dict,
                "graph_edges_relevances": graph_edges_relevances_dict}

