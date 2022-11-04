"""
    GNN actions - Predict & Retrain
    Dataset contains only homogeneous actions

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
from numpy.random import RandomState
import torch
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear

from gnns.gnns_graph_classification.GCN_Graph_Classification import GCN
from gnns.gnns_graph_classification.gnn_train_test_methods import train_model, use_trained_model
from utils.gnn_load_save import load_gnn_model, save_gnn_model


class GNN_Actions(torch.nn.Module):
    """
    GNN Actions
    """

    def __init__(self, gnn_architecture_params_dict: dict, dataset_name: str):
        """
        Init the GNN actions

        :param gnn_architecture_params_dict: Dictionary of architecture parameters
        :param dataset_name: Dataset name
        """

        super(GNN_Actions, self).__init__()

        # [1.] Data splitting parameters -------------------------------------------------------------------------------
        self.proportion_of_training_set = 3/4
        self.train_dataset_shuffled_indexes = None
        self.test_dataset_shuffled_indexes = None

        # [2.] GNN architecture parameters -----------------------------------------------------------------------------
        self.gnn_architecture_params_dict = gnn_architecture_params_dict
        self.dataset_name = dataset_name

        # [3.] GNN training parameters ---------------------------------------------------------------------------------
        self.batch_size = 8
        self.epochs_nr = 30

        # [4.] Data structures for the performance metrics -------------------------------------------------------------
        self.train_set_metrics_dict = None
        self.train_outputs_predictions_dict = None
        self.test_set_metrics_dict = None
        self.test_outputs_predictions_dict = None

        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # [5.] Init the parameters of the layers -----------------------------------------------------------------------
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        num_features = 2
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = 'cuda:0'
        num_classes = 2
        model = GCN(
            num_node_features=num_features,
            hidden_channels=self.gnn_architecture_params_dict["hidden_channels"],
            layers_nr=self.gnn_architecture_params_dict["layers_nr"],
            num_classes=num_classes). \
            to(device)

        prng = RandomState(1234567890)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # GCNConv layers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.predefined_weights_dict = {}

        for layer in model.conv_layers_list:
            if hasattr(layer, 'reset_parameters'):

                # Weight and bias of the GCNConv layers ------------------------------------------------------------
                lin_weight_data_size = layer.lin.weight.data.size()
                lin_weight_data_random_weights = prng.uniform(-1.0,
                                                              1.0,
                                                              (lin_weight_data_size[0], lin_weight_data_size[1]))
                self.predefined_weights_dict[str(layer) + "_lin_weight"] = \
                    torch.from_numpy(lin_weight_data_random_weights).to(dtype=torch.float32)

                # Bias of the GCNConv layers -----------------------------------------------------------------------
                layer_bias_data_size = layer.bias.size()
                layer_bias_data_random_weights = prng.uniform(-1.0,
                                                              1.0,
                                                              (layer_bias_data_size[0],))
                self.predefined_weights_dict[str(layer) + "_bias"] = \
                    torch.from_numpy(layer_bias_data_random_weights).to(dtype=torch.float32)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Linear layer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):

                # Weight and bias of the Linear layer --------------------------------------------------------------
                layer_weight_data_size = layer.weight.data.size()
                layer_weight_data_random_weights = prng.uniform(-1.0,
                                                                1.0,
                                                                (layer_weight_data_size[0],
                                                                 layer_weight_data_size[1]))
                self.predefined_weights_dict[str(layer) + "_weight"] = \
                    torch.from_numpy(layer_weight_data_random_weights).to(dtype=torch.float32)

                # Bias of the Linear layers ------------------------------------------------------------------------
                layer_bias_data_size = layer.bias.data.size()
                layer_bias_data_random_weights = prng.uniform(-1.0,
                                                              1.0,
                                                              (layer_bias_data_size[0],))
                self.predefined_weights_dict[str(layer) + "_bias"] = \
                    torch.from_numpy(layer_bias_data_random_weights).to(dtype=torch.float32)

    def is_in_training_set(self, graph_id) -> bool:
        """
        Check if the graph is in the training or test set

        :param graph_id: Id of graph

        :return: Boolean value indicating if this graph is in the training set
        """

        b_is_in_train = True

        graph_numbering_ids = re.findall('[0-9]+', graph_id)
        graph_nr_in_dataset = int(graph_numbering_ids[0])

        if self.test_dataset_shuffled_indexes is None:
            gnn_model_info = load_gnn_model(self.dataset_name, True)
            self.test_dataset_shuffled_indexes = gnn_model_info["test_dataset_shuffled_indexes"]

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

        :return: Tuple of model and dictionary of performance metrics
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
        model = GCN(
            num_node_features=num_features,
            hidden_channels=self.gnn_architecture_params_dict["hidden_channels"],
            layers_nr=self.gnn_architecture_params_dict["layers_nr"],
            num_classes=num_classes).\
            to(device)
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
                # print('Trigger times reset to 0.')
                trigger_times = 0

            last_accuracy = current_test_set_accuracy

        # [5.] Print the end test set metrics --------------------------------------------------------------------------
        # print(self.test_set_metrics_dict)

        ################################################################################################################
        # [6.] GNN store ===============================================================================================
        ################################################################################################################
        save_gnn_model(model,
                       self.train_set_metrics_dict, self.test_set_metrics_dict,
                       self.train_dataset_shuffled_indexes, self.test_dataset_shuffled_indexes,
                       self.train_outputs_predictions_dict["prediction_confidences"],
                       self.test_outputs_predictions_dict["prediction_confidences"],
                       self.train_outputs_predictions_dict["output_classes"],
                       self.test_outputs_predictions_dict["output_classes"],
                       self.dataset_name,
                       True)

        return model, self.test_set_metrics_dict

    def gnn_predict(self, model: GCN, input_graph: Data, user_token: str) -> tuple:
        """
        GNN predict function.

        :param model: The GNN model
        :param input_graph: The input graph that we need its prediction
        :param user_token: Identifies a user

        [1.] Load the GNN from the file system.
        [2.] Make the check if the (generally) changed input graphs conform to the architecture of the loaded GNN.
             If the requirements are not fulfilled, then the predict function cannot be applied.
        [3.] Apply the predict function on the GNN

        :return:
        """

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = 'cuda:0'

        user_token = str(user_token)

        ################################################################################################################
        # [0.] Preprocessing ===========================================================================================
        ################################################################################################################
        input_graph.to(device)

        ################################################################################################################
        # [1.] Make the check if the (generally) changed input graphs conform to the architecture of the loaded GNN  ===
        ################################################################################################################
        model_state_dict = model.state_dict()
        input_features_model_nr = model_state_dict["gcns_modules.0.lin.weight"].size(dim=1)
        input_features_graph_nr = input_graph.x.size(dim=1)
        assert input_features_model_nr == input_features_graph_nr, \
            f"The number of features of the nodes that the model expects {input_features_model_nr},\n" \
            f"is unequal to the actual number of features of the nodes in the changed graph: {input_features_graph_nr}."

        ################################################################################################################
        # [2.] Predict =================================================================================================
        ################################################################################################################
        testing_graph_list = [input_graph]
        testing_graph_loader = DataLoader(testing_graph_list, batch_size=1, shuffle=False)

        for data in testing_graph_loader:
            output_of_testing = model(data.x, data.edge_index, data.batch).cpu().detach().numpy()[0]

            prediction_label_of_testing = np.argmax(output_of_testing)
            prediction_confidence_of_testing = np.amax(output_of_testing)

        return str(prediction_label_of_testing), str(round(prediction_confidence_of_testing, 2))

    def gnn_retrain(self, model: GCN, input_graphs: list, user_token: str) -> tuple:
        """
        GNN retrain function

        :param model: The GNN model
        :param input_graphs: The input graphs that we need its prediction
        :param user_token: Identifies a user

        [1.] Get the architectural characteristics of the GNN in the file system - you don't have to load it.
        [2.] Make the check if the (generally) changed input graphs conform to the architecture of the stored GNN.
             If the requirements are not fulfilled, then the predict function cannot be applied.
        [3.] Apply the predict function on the GNN

        :return: Tuple of model and dictionary of performance metrics
        """

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = 'cuda:0'

        user_token = str(user_token)

        ################################################################################################################
        # [0.] Preprocessing - normalization ===========================================================================
        ################################################################################################################
        # normalized_graphs_dataset = graph_features_normalization(input_graphs)

        for graph in input_graphs:
            graph.to(device)

        if self.train_dataset_shuffled_indexes is None or self.test_dataset_shuffled_indexes is None:
            gnn_model_info = load_gnn_model(self.dataset_name, False, user_token)
            self.train_dataset_shuffled_indexes = gnn_model_info["train_dataset_shuffled_indexes"]
            self.test_dataset_shuffled_indexes = gnn_model_info["test_dataset_shuffled_indexes"]

        train_normalized_graphs_dataset = [input_graphs[idx]
                                           for idx in self.train_dataset_shuffled_indexes]
        test_normalized_graphs_dataset = [input_graphs[idx] for idx in self.test_dataset_shuffled_indexes]

        re_train_loader = DataLoader(train_normalized_graphs_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False)      # Training set typically has to be shuffled -------------------
        re_test_loader = DataLoader(test_normalized_graphs_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=False)      # Test set typically has to be the same (not shuffled) ---------

        ################################################################################################################
        # [1.] Make the check if the (generally) changed input graphs conform to the architecture of the loaded GNN ====
        #      Use the first graph for now =============================================================================
        ################################################################################################################
        model_state_dict = model.state_dict()
        input_features_model_nr = model_state_dict["gcns_modules.0.lin.weight"].size(dim=1)
        input_features_graph_nr = input_graphs[0].x.size(dim=1)

        # [1.1.] If the number of features did not change, then just reset the weights ---------------------------------
        if input_features_model_nr == input_features_graph_nr:

            prng = RandomState(1234567890)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # GCNConv layers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for layer in model.conv_layers_list:

                if hasattr(layer, 'reset_parameters'):
                    # layer.reset_parameters()          # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                    # Weight and bias of the GCNConv layers ------------------------------------------------------------
                    # lin_weight_data_size = layer.lin.weight.data.size()
                    # lin_weight_data_random_weights = prng.uniform(-1.0,
                    #                                              1.0,
                    #                                              (lin_weight_data_size[0], lin_weight_data_size[1]))
                    # layer.lin.weight.data = torch.from_numpy(lin_weight_data_random_weights).to(dtype=torch.float32)
                    layer.lin.weight.data = self.predefined_weights_dict[str(layer) + "_lin_weight"]

                    # Bias of the GCNConv layers -----------------------------------------------------------------------
                    # layer_bias_data_size = layer.bias.size()
                    # layer_bias_data_random_weights = prng.uniform(-1.0,
                    #                                              1.0,
                    #                                              (layer_bias_data_size[0], ))
                    # layer.bias.data = torch.from_numpy(layer_bias_data_random_weights).to(dtype=torch.float32)
                    layer.bias.data = self.predefined_weights_dict[str(layer) + "_bias"]

                    layer.to(device)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Linear layer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for layer in model.children():

                if hasattr(layer, 'reset_parameters'):
                    # layer.reset_parameters()          # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                    # Weight and bias of the Linear layer --------------------------------------------------------------
                    # layer_weight_data_size = layer.weight.data.size()
                    # layer_weight_data_random_weights = prng.uniform(-1.0,
                    #                                                1.0,
                    #                                                (layer_weight_data_size[0],
                    #                                                 layer_weight_data_size[1]))
                    # layer.weight.data = torch.from_numpy(layer_weight_data_random_weights).to(dtype=torch.float32)
                    layer.weight.data = self.predefined_weights_dict[str(layer) + "_weight"]

                    # Bias of the Linear layers ------------------------------------------------------------------------
                    # layer_bias_data_size = layer.bias.data.size()
                    # layer_bias_data_random_weights = prng.uniform(-1.0,
                    #                                              1.0,
                    #                                              (layer_bias_data_size[0], ))
                    # layer.bias.data = torch.from_numpy(layer_bias_data_random_weights).to(dtype=torch.float32)
                    layer.bias.data = self.predefined_weights_dict[str(layer) + "_bias"]

                    layer.to(device)

        # [1.2.] If the number of features did change, you need a new architecture -------------------------------------
        else:
            print(f"The number of features of the nodes that the model expects {input_features_model_nr},\n"
                  f"is unequal to the actual number of features of the nodes in the changed graph: "
                  f"{input_features_graph_nr}.")

            num_classes = model_state_dict["lin.weight"].size(dim=0)
            hidden_channels = model_state_dict["gcns_modules.0.lin.weight"].size(dim=0)
            model = GCN(num_node_features=input_features_graph_nr,
                        hidden_channels=hidden_channels,
                        num_classes=num_classes).to(device)

        ################################################################################################################
        # [2.] Retrain =================================================================================================
        ################################################################################################################
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        last_accuracy = sys.float_info.min
        patience = 5
        trigger_times = 0

        # [3.] Iterate over several epochs -----------------------------------------------------------------------------
        for epoch in range(1, self.epochs_nr + 1):

            # print(f"Epoch: {epoch}")

            # [4.] Retrain the model and gather the performance metrics of the training and test set -------------------
            train_model(model, re_train_loader, optimizer, criterion)
            self.train_set_metrics_dict, self.train_outputs_predictions_dict = \
                use_trained_model(model, re_train_loader)
            self.test_set_metrics_dict, self.test_outputs_predictions_dict = use_trained_model(model, re_test_loader)

            # [5.] Apply early stopping --------------------------------------------------------------------------------
            current_test_set_accuracy = float(self.test_set_metrics_dict['accuracy'])

            if current_test_set_accuracy < last_accuracy:
                trigger_times += 1

                if trigger_times >= patience:
                    print(f'Early stopping at epoch {epoch}.')
                    break

            else:
                # print('Trigger times reset to 0.')
                trigger_times = 0

            last_accuracy = current_test_set_accuracy

        ################################################################################################################
        # [6.] GNN store ===============================================================================================
        ################################################################################################################
        print(f"Train set performance: {self.train_set_metrics_dict}")
        print(f"Test set performance: {self.test_set_metrics_dict}")

        save_gnn_model(model,
                       self.train_set_metrics_dict, self.test_set_metrics_dict,
                       self.train_dataset_shuffled_indexes, self.test_dataset_shuffled_indexes,
                       self.train_outputs_predictions_dict["prediction_confidences"],
                       self.test_outputs_predictions_dict["prediction_confidences"],
                       self.train_outputs_predictions_dict["output_classes"],
                       self.test_outputs_predictions_dict["output_classes"],
                       self.dataset_name,
                       False,
                       user_token)

        ################################################################################################################
        # [7.] Return the new test set metrics after re-train ==========================================================
        ################################################################################################################
        return model, self.test_set_metrics_dict
