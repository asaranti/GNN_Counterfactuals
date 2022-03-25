"""
    GNN actions - Predict & Retrain

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-02-18
"""

from operator import itemgetter
import os
import random

from sklearn.preprocessing import minmax_scale
import torch
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader

from gnns.gnns_graph_classification.GCN_Graph_Classification import GCN
from gnns.gnns_graph_classification.gnn_train_test_methods import train, test


def gnn_init_train(original_dataset: list):
    """
    Method that implements the first training of the GNN

    :param original_dataset: Original dataset - List of graphs
    """

    ####################################################################################################################
    # [0.] Data Preparation ============================================================================================
    ####################################################################################################################
    # [0.0.] Choose device ---------------------------------------------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = 'cuda:0'

    # [0.1.] Input features preprocessing/normalization ----------------------------------------------------------------
    for graph in original_dataset:
        x_features = graph.x
        x_features_array = x_features.cpu().detach().numpy()

        x_features_transformed = minmax_scale(x_features_array, feature_range=(0, 1))
        graph.x = torch.tensor(x_features_transformed)
        graph.to(device)

    # [0.2.] Split training/validation/test set ------------------------------------------------------------------------
    graph_0 = original_dataset[0]
    num_features = graph_0.num_node_features
    graphs_nr = len(original_dataset)

    # [0.3.] Shuffle the dataset and keep the list indexes -------------------------------------------------------------
    x = list(enumerate(original_dataset))
    random.shuffle(x)
    random_indices, graphs_list = zip(*x)
    dataset_random_shuffling = list(itemgetter(*random_indices)(original_dataset))

    # [0.4.] Split to training and test set ----------------------------------------------------------------------------
    train_dataset_len = int(graphs_nr * 3 / 4)
    train_dataset = dataset_random_shuffling[:train_dataset_len]
    test_dataset = dataset_random_shuffling[train_dataset_len:]

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ####################################################################################################################
    # [1.] Graph Classification ========================================================================================
    ####################################################################################################################
    num_classes = 2
    model = GCN(num_node_features=num_features, hidden_channels=200, num_classes=num_classes).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    epochs_nr = 20
    for epoch in range(1, epochs_nr + 1):
        train(model, train_loader, optimizer, criterion)
        train_set_metrics_dict = test(model, train_loader)
        test_set_metrics_dict = test(model, test_loader)

    test_set_metrics_dict = test(model, test_loader)
    print(test_set_metrics_dict)

    ####################################################################################################################
    # [2.] GNN store ===================================================================================================
    ####################################################################################################################
    gnn_storage_folder = os.path.join("data", "output", "gnns")
    gnn_model_file_path = os.path.join(gnn_storage_folder, "gcn_model.pth")
    torch.save(model, gnn_model_file_path)

    return test_set_metrics_dict


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
