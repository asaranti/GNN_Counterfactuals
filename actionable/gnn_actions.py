"""
    GNN actions - Predict & Retrain

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-02-18
"""

import os

import torch
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader

from gnns.gnns_graph_classification.GCN_Graph_Classification import GCN
from gnns.gnns_graph_classification.gnn_train_test_methods import train, test


def gnn_predict(input_graph: Data):
    """
    GNN predict function.

    [1.] Load the GNN from the file system.
    [2.] Make the check if the (generally) changed input graphs conform to the architecture of the loaded GNN.
         If the requirements are not fulfilled, then the predict function cannot be applied.
    [3.] Apply the predict function on the GNN
    """

    ####################################################################################################################
    # [1.] Load the GNN from the file system ===========================================================================
    ####################################################################################################################
    gnn_storage_folder = os.path.join("data", "output", "gnns")
    gnn_model_file_path = os.path.join(gnn_storage_folder, "gcn_model.pth")
    model = torch.load(gnn_model_file_path)
    model.eval()

    ####################################################################################################################
    # [2.] Make the check if the (generally) changed input graphs conform to the architecture of the loaded GNN  =======
    ####################################################################################################################
    model_state_dict = model.state_dict()
    input_features_model_nr = model_state_dict["conv1.lin.weight"].size(dim=1)
    input_features_graph_nr = input_graph.x.size(dim=1)
    assert input_features_model_nr == input_features_graph_nr, \
        f"The number of features of the nodes that the model expects {input_features_model_nr},\n" \
        f"is unequal to the actual number of features of the nodes in the changed graph: {input_features_graph_nr}."

    ####################################################################################################################
    # [3.] Predict =====================================================================================================
    ####################################################################################################################
    testing_graph_list = [input_graph]
    testing_graph_loader = DataLoader(testing_graph_list, batch_size=1, shuffle=False)
    for data in testing_graph_loader:
        output_of_testing = model.forward(data.x, data.edge_index, data.batch)
        prediction_of_testing = output_of_testing.argmax(dim=1)
        print(f"Prediction of testing: {prediction_of_testing}")


def gnn_retrain(input_graphs: list):
    """
    GNN retrain function

    [1.] Get the architectural characteristics of the GNN in the file system - you don't have to load it.
    [2.] Make the check if the (generally) changed input graphs conform to the architecture of the stored GNN.
         If the requirements are not fulfilled, then the predict function cannot be applied.
    [3.] Apply the predict function on the GNN
    """

    ####################################################################################################################
    # [1.] Load the GNN, get the architecture ==========================================================================
    ####################################################################################################################
    gnn_storage_folder = os.path.join("data", "output", "gnns")
    gnn_model_file_path = os.path.join(gnn_storage_folder, "gcn_model.pth")
    model = torch.load(gnn_model_file_path)
    model.eval()

    ####################################################################################################################
    # [2.] Make the check if the (generally) changed input graphs conform to the architecture of the loaded GNN ========
    #      Use the first graph for now =================================================================================
    ####################################################################################################################
    model_state_dict = model.state_dict()
    input_features_model_nr = model_state_dict["conv1.lin.weight"].size(dim=1)
    input_features_graph_nr = input_graphs[0].x.size(dim=1)

    # [2.1.] If the number of features did not change, then just reset the weights -------------------------------------
    if input_features_model_nr == input_features_graph_nr:
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    # [2.2.] If the number of features did change, you need a new architecture -----------------------------------------
    else:
        print(f"The number of features of the nodes that the model expects {input_features_model_nr},\n"
              f"is unequal to the actual number of features of the nodes in the changed graph: "
              f"{input_features_graph_nr}.")

        num_classes = model_state_dict["lin.weight"].size(dim=0)
        hidden_channels = model_state_dict["conv1.lin.weight"].size(dim=0)
        model = GCN(num_node_features=input_features_graph_nr,
                    hidden_channels=hidden_channels,
                    num_classes=num_classes)

    ####################################################################################################################
    # [3.] Retrain =====================================================================================================
    ####################################################################################################################
    batch_size = 8
    epochs_nr = 20

    re_train_dataset = DataLoader(input_graphs, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, epochs_nr + 1):
        print(f"Epoch: {epoch}")

        train(model, re_train_dataset, optimizer, criterion)
        train_acc = test(model, re_train_dataset)

        print(f'Epoch: {epoch:03d}, Tra+in Acc: {train_acc:.4f}')
        print("-------------------------------------------------------------------------")
