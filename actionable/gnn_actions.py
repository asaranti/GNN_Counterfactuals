"""
    GNN actions - Predict & Retrain

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-02-18
"""

from operator import itemgetter
import os
import random
import re
import sys

import numpy as np
import torch
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader

from gnns.gnns_graph_classification.GCN_Graph_Classification import GCN
from gnns.gnns_graph_classification.gnn_train_test_methods import train_model, use_trained_model
from preprocessing_data.graph_features_normalization import graph_features_normalization


class GNN_Actions(torch.nn.Module):
    """
    GNN Actions
    """

    def __init__(self):
        """
        Init
        """

        super(GNN_Actions, self).__init__()

        # [1.] Data splitting parameters -------------------------------------------------------------------------------
        self.proportion_of_training_set = 3/4
        self.train_dataset_shuffled_indexes = None
        self.test_dataset_shuffled_indexes = None

        # [2.] GNN training parameters ---------------------------------------------------------------------------------
        self.batch_size = 8
        self.epochs_nr = 100

        # [3.] Data structures for the performance metrics -------------------------------------------------------------
        self.train_set_metrics_dict = None
        self.train_outputs_predictions_dict = None
        self.test_set_metrics_dict = None
        self.test_outputs_predictions_dict = None

    def is_in_training_set(self, graph_id) -> bool:
        """
        Check if the graph is in the training or test set

        :param graph_id: Id of graph

        :return: Boolean value indicating if this graph is in the training set
        """

        b_is_in_train = True

        graph_numbering_ids = re.findall('[0-9]+', graph_id)
        graph_nr_in_dataset = int(graph_numbering_ids[0])

        if graph_nr_in_dataset in self.test_dataset_shuffled_indexes:
            b_is_in_train = False

        return b_is_in_train

    def gnn_init_preprocessing(self, original_dataset: list):
        """
        Method GNN preprocessing_files: Make the initial split of the dataset and store the indexes of the split

        :param original_dataset: Original dataset - List of graphs
        """

        graphs_nr = len(original_dataset)

        # [1.] Shuffle the dataset and keep the list indexes -----------------------------------------------------------
        x = list(enumerate(original_dataset))
        random.shuffle(x)
        random_indices, graphs_list = zip(*x)
        dataset_random_shuffling = list(itemgetter(*random_indices)(original_dataset))

        # [2.] Split to training and test set --------------------------------------------------------------------------
        train_dataset_len = int(graphs_nr * self.proportion_of_training_set)
        train_dataset = dataset_random_shuffling[:train_dataset_len]
        test_dataset = dataset_random_shuffling[train_dataset_len:]
        self.train_dataset_shuffled_indexes = random_indices[:train_dataset_len]
        self.test_dataset_shuffled_indexes = random_indices[train_dataset_len:]

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def gnn_init_train(self, input_graphs: list) -> dict:
        """
        Method that implements the first training of the GNN

        :param input_graphs: Original dataset - List of graphs
        """

        ################################################################################################################
        # [0.] Data Preparation ========================================================================================
        ################################################################################################################
        # [0.0.] Choose device -----------------------------------------------------------------------------------------
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = 'cuda:0'
        # device = 'cpu'

        # [0.1.] Input features preprocessing_files/normalization ------------------------------------------------------
        graphs_nr = len(input_graphs)

        for graph in input_graphs:
            graph.to(device)

        # normalized_graphs_dataset = graph_features_normalization(input_graphs)
        for graph in input_graphs:

            x_features_array = graph.x.cpu().detach().numpy()
            graph.x = torch.tensor(x_features_array).to(dtype=torch.float32)    # float32 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            graph.to(device)
        # input_graphs = normalized_graphs_dataset

        # [0.2.] Split training/validation/test set --------------------------------------------------------------------
        graph_0 = input_graphs[0]
        num_features = graph_0.num_node_features

        # [0.3.] Shuffle the dataset and keep the list indexes ---------------------------------------------------------
        # [0.4.] Split to training and test set ------------------------------------------------------------------------
        train_loader, test_loader = self.gnn_init_preprocessing(input_graphs)

        ################################################################################################################
        # [1.] Graph Classification ====================================================================================
        ################################################################################################################
        num_classes = 2
        model = GCN(num_node_features=num_features, hidden_channels=100, num_classes=num_classes).to(device)
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        last_accuracy = sys.float_info.min
        patience = 5
        trigger_times = 0

        # [2.] Iterate over several epochs -----------------------------------------------------------------------------
        for epoch in range(1, self.epochs_nr + 1):

            print(f"Epoch: {epoch}")

            # [3.] Train the model and gather the performance metrics of the training and test set ---------------------
            train_model(model, train_loader, optimizer, criterion)

            self.train_set_metrics_dict, self.train_outputs_predictions_dict = use_trained_model(model, train_loader)
            self.test_set_metrics_dict, self.test_outputs_predictions_dict = use_trained_model(model, test_loader)

            current_test_set_accuracy = float(self.test_set_metrics_dict['accuracy'])

            # [4.] Apply early stopping --------------------------------------------------------------------------------
            if current_test_set_accuracy < last_accuracy:
                trigger_times += 1

                if trigger_times >= patience:
                    print(f'Early stopping at epoch {epoch}.')
                    break

            else:
                print('Trigger times reset to 0.')
                trigger_times = 0

            last_accuracy = current_test_set_accuracy

        # [5.] Print the end test set metrics --------------------------------------------------------------------------
        # print(self.test_set_metrics_dict)

        ################################################################################################################
        # [6.] GNN store ===============================================================================================
        ################################################################################################################
        gnn_storage_folder = os.path.join("data", "output", "gnns")
        gnn_model_file_path = os.path.join(gnn_storage_folder, "gcn_model.pth")
        torch.save(model, gnn_model_file_path)

        return self.test_set_metrics_dict

    def gnn_predict(self, input_graph: Data) -> tuple:
        """
        GNN predict function.

        :param input_graph: The input graph that we need its prediction

        [1.] Load the GNN from the file system.
        [2.] Make the check if the (generally) changed input graphs conform to the architecture of the loaded GNN.
             If the requirements are not fulfilled, then the predict function cannot be applied.
        [3.] Apply the predict function on the GNN

        :return:
        """

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = 'cuda:0'

        ################################################################################################################
        # [0.] Preprocessing ===========================================================================================
        ################################################################################################################
        input_graph.to(device)

        ################################################################################################################
        # [1.] Load the GNN from the file system =======================================================================
        ################################################################################################################
        gnn_storage_folder = os.path.join("data", "output", "gnns")
        gnn_model_file_path = os.path.join(gnn_storage_folder, "gcn_model.pth")
        model = torch.load(gnn_model_file_path).to(device)
        model.eval()

        ################################################################################################################
        # [2.] Make the check if the (generally) changed input graphs conform to the architecture of the loaded GNN  ===
        ################################################################################################################
        model_state_dict = model.state_dict()
        input_features_model_nr = model_state_dict["conv1.lin.weight"].size(dim=1)
        input_features_graph_nr = input_graph.x.size(dim=1)
        assert input_features_model_nr == input_features_graph_nr, \
            f"The number of features of the nodes that the model expects {input_features_model_nr},\n" \
            f"is unequal to the actual number of features of the nodes in the changed graph: {input_features_graph_nr}."

        ################################################################################################################
        # [3.] Predict =================================================================================================
        ################################################################################################################
        testing_graph_list = [input_graph]
        testing_graph_loader = DataLoader(testing_graph_list, batch_size=1, shuffle=False)

        for data in testing_graph_loader:
            output_of_testing = model(data.x, data.edge_index, data.batch).cpu().detach().numpy()[0]

            prediction_label_of_testing = np.argmax(output_of_testing)
            prediction_confidence_of_testing = np.amax(output_of_testing)

        return str(prediction_label_of_testing), str(round(prediction_confidence_of_testing, 2))

    def gnn_retrain(self, input_graphs: list) -> dict:
        """
        GNN retrain function

        [1.] Get the architectural characteristics of the GNN in the file system - you don't have to load it.
        [2.] Make the check if the (generally) changed input graphs conform to the architecture of the stored GNN.
             If the requirements are not fulfilled, then the predict function cannot be applied.
        [3.] Apply the predict function on the GNN
        """

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = 'cuda:0'

        ################################################################################################################
        # [0.] Preprocessing - normalization ===========================================================================
        ################################################################################################################
        # normalized_graphs_dataset = graph_features_normalization(input_graphs)

        for graph in input_graphs:
            graph.to(device)

        train_normalized_graphs_dataset = [input_graphs[idx]
                                           for idx in self.train_dataset_shuffled_indexes]
        test_normalized_graphs_dataset = [input_graphs[idx] for idx in self.test_dataset_shuffled_indexes]

        re_train_loader = DataLoader(train_normalized_graphs_dataset, batch_size=self.batch_size, shuffle=True)
        re_test_loader = DataLoader(test_normalized_graphs_dataset, batch_size=self.batch_size, shuffle=False)

        ################################################################################################################
        # [1.] Load the GNN, get the architecture ======================================================================
        ################################################################################################################
        gnn_storage_folder = os.path.join("data", "output", "gnns")
        gnn_model_file_path = os.path.join(gnn_storage_folder, "gcn_model.pth")
        model = torch.load(gnn_model_file_path).to(device)
        model.eval()

        ################################################################################################################
        # [2.] Make the check if the (generally) changed input graphs conform to the architecture of the loaded GNN ====
        #      Use the first graph for now =============================================================================
        ################################################################################################################
        model_state_dict = model.state_dict()
        input_features_model_nr = model_state_dict["conv1.lin.weight"].size(dim=1)
        input_features_graph_nr = input_graphs[0].x.size(dim=1)

        # [2.1.] If the number of features did not change, then just reset the weights ---------------------------------
        if input_features_model_nr == input_features_graph_nr:
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        # [2.2.] If the number of features did change, you need a new architecture -------------------------------------
        else:
            print(f"The number of features of the nodes that the model expects {input_features_model_nr},\n"
                  f"is unequal to the actual number of features of the nodes in the changed graph: "
                  f"{input_features_graph_nr}.")

            num_classes = model_state_dict["lin.weight"].size(dim=0)
            hidden_channels = model_state_dict["conv1.lin.weight"].size(dim=0)
            model = GCN(num_node_features=input_features_graph_nr,
                        hidden_channels=hidden_channels,
                        num_classes=num_classes).to(device)

        ################################################################################################################
        # [3.] Retrain =================================================================================================
        ################################################################################################################
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        last_accuracy = sys.float_info.min
        patience = 5
        trigger_times = 0

        # [4.] Iterate over several epochs -----------------------------------------------------------------------------
        for epoch in range(1, self.epochs_nr + 1):

            print(f"Epoch: {epoch}")

            # [5.] Retrain the model and gather the performance metrics of the training and test set -------------------
            train_model(model, re_train_loader, optimizer, criterion)
            self.train_set_metrics_dict, self.train_outputs_predictions_dict = \
                use_trained_model(model, re_train_loader)
            self.test_set_metrics_dict, self.test_outputs_predictions_dict = use_trained_model(model, re_test_loader)

            # [6.] Apply early stopping --------------------------------------------------------------------------------
            current_test_set_accuracy = float(self.test_set_metrics_dict['accuracy'])

            if current_test_set_accuracy < last_accuracy:
                trigger_times += 1

                if trigger_times >= patience:
                    print(f'Early stopping at epoch {epoch}.')
                    break

            else:
                print('Trigger times reset to 0.')
                trigger_times = 0

            last_accuracy = current_test_set_accuracy

        ################################################################################################################
        # [7.] GNN store ===============================================================================================
        ################################################################################################################
        gnn_storage_folder = os.path.join("data", "output", "gnns")
        gnn_model_file_path = os.path.join(gnn_storage_folder, "gcn_model.pth")
        torch.save(model, gnn_model_file_path)

        ################################################################################################################
        # [8.] Return the new test set metrics after re-train ==========================================================
        ################################################################################################################
        return self.test_set_metrics_dict
