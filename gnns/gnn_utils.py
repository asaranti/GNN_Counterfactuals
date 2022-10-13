"""
    GNN utilities - Save and restore the model

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-10-13
"""

import os
import pickle

import torch

from gnns.gnns_graph_classification.GCN_Graph_Classification import GCN


def load_gnn_model(dataset_name: str) -> dict:
    """
    Load all model-relevant information from local storage

    [1.] The model in ".pth" format
    [2.] The overall performance of the model (confusion matrix, sensitivity, specificity)
    [3.] Two lists of indexes of the graph ids for the training and the test set
    [4.] The exact prediction confidence for each graph in the dataset
    [5.] The predicted class for each graph in the dataset

    :param dataset_name: Dataset name that specifies the name of the model too
    :return: Dictionary of all the restored data
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = 'cuda:0'

    gnn_storage_folder = os.path.join("models", dataset_name)

    # [1.] The model in ".pth" format ----------------------------------------------------------------------------------
    model = torch.load(os.path.join(gnn_storage_folder, f"{dataset_name}_model.pth")).to(device)
    model.eval()

    # [2.] The overall performance of the model (confusion matrix, sensitivity, specificity) ---------------------------
    train_set_metrics_dict = pickle.load(open(os.path.join(gnn_storage_folder, 'train_set_metrics_dict.pkl'), "rb"))
    test_set_metrics_dict = pickle.load(open(os.path.join(gnn_storage_folder, 'test_set_metrics_dict.pkl'), "rb"))

    # [3.] Two lists of indexes of the graph ids for the training and the test set -------------------------------------
    train_dataset_shuffled_indexes = pickle.load(open(os.path.join(gnn_storage_folder,
                                                                   'train_dataset_shuffled_indexes.pkl'), "rb"))
    test_dataset_shuffled_indexes = pickle.load(open(os.path.join(gnn_storage_folder,
                                                                  'test_dataset_shuffled_indexes.pkl'), "rb"))

    # [4.] Load the exact prediction confidence for each graph in the dataset ------------------------------------------
    train_prediction_confidence_dict = pickle.load(open(os.path.join(gnn_storage_folder,
                                                   'train_prediction_confidence_dict.pkl'), "rb"))
    test_prediction_confidence_dict = pickle.load(open(os.path.join(gnn_storage_folder,
                                                  'test_prediction_confidence_dict.pkl'), "rb"))

    # [5.] Load the output class for each graph in the dataset  --------------------------------------------------------
    train_output_class_dict = pickle.load(open(os.path.join(gnn_storage_folder,
                                          'train_output_class_dict.pkl'), "rb"))
    test_output_class_dict = pickle.load(open(os.path.join(gnn_storage_folder,
                                         'test_output_class_dict.pkl'), "rb"))

    return {"model": model,
            "train_set_metrics_dict": train_set_metrics_dict,
            "test_set_metrics_dict": test_set_metrics_dict,
            "train_dataset_shuffled_indexes": train_dataset_shuffled_indexes,
            "test_dataset_shuffled_indexes": test_dataset_shuffled_indexes,
            "train_prediction_confidence_dict": train_prediction_confidence_dict,
            "test_prediction_confidence_dict": test_prediction_confidence_dict,
            "train_outputs_predictions_dict": train_output_class_dict,
            "test_outputs_predictions_dict": test_output_class_dict
            }


def save_gnn_model(model: GCN,
                   train_set_metrics_dict: dict, test_set_metrics_dict: dict,
                   train_dataset_shuffled_indexes: list, test_dataset_shuffled_indexes: list,
                   train_prediction_confidence_dict: dict, test_prediction_confidence_dict: dict,
                   train_output_class_dict: dict, test_output_class_dict: dict,
                   dataset_name: str):
    """
    Save all model-relevant information locally.
    They will be needed for further retraining or presentation purposes.

    [1.] The model in ".pth" format
    [2.] The overall performance of the model (confusion matrix, sensitivity, specificity)
    [3.] Two lists of indexes of the graph ids for the training and the test set
    [4.] The exact prediction confidence for each graph in the dataset
    [5.] The predicted class for each graph in the dataset

    :param model: Model to be saved
    :param train_set_metrics_dict: Performance of the model (confusion matrix, sensitivity, specificity)
                                   on the training set
    :param test_set_metrics_dict: Performance of the model (confusion matrix, sensitivity, specificity)
                                  on the test set
    :param train_dataset_shuffled_indexes: Indexes of the graphs that are used in the training set
    :param test_dataset_shuffled_indexes: Indexes of the graphs that are used in the test set
    :param train_prediction_confidence_dict: Prediction confidence for each graph in the training set
    :param test_prediction_confidence_dict: Prediction confidence for each graph in the test set
    :param train_output_class_dict: The predicted classes of the training set graphs
    :param test_output_class_dict: The predicted classes of the test set graphs
    :param dataset_name: Dataset name that specifies the name of the model too
    """

    # [0.] Create the folder of storing the GNN-relevant information ---------------------------------------------------
    gnn_storage_folder = os.path.join("models", dataset_name)
    if not os.path.exists(gnn_storage_folder):
        os.makedirs(gnn_storage_folder)

    # [1.] Save the model in ".pth" format -----------------------------------------------------------------------------
    gnn_model_file_path = os.path.join(gnn_storage_folder, f"{dataset_name}_model.pth")
    torch.save(model, gnn_model_file_path)

    # [2.] Save the overall performance of the model (confusion matrix, sensitivity, specificity) ----------------------
    #      for the training set and the test set [pickle] --------------------------------------------------------------
    with open(os.path.join(gnn_storage_folder, 'train_set_metrics_dict.pkl'), 'wb') as f:
        pickle.dump(train_set_metrics_dict, f)
    with open(os.path.join(gnn_storage_folder, 'test_set_metrics_dict.pkl'), 'wb') as f:
        pickle.dump(test_set_metrics_dict, f)

    # [3.] Save the two lists of indexes of the graph ids for the training and the test set ----------------------------
    with open(os.path.join(gnn_storage_folder, 'train_dataset_shuffled_indexes.pkl'), 'wb') as f:
        pickle.dump(train_dataset_shuffled_indexes, f)
    with open(os.path.join(gnn_storage_folder, 'test_dataset_shuffled_indexes.pkl'), 'wb') as f:
        pickle.dump(test_dataset_shuffled_indexes, f)

    # [4.] Save the exact prediction confidence for each graph in the dataset ------------------------------------------
    with open(os.path.join(gnn_storage_folder, 'train_prediction_confidence_dict.pkl'), 'wb') as f:
        pickle.dump(train_prediction_confidence_dict, f)
    with open(os.path.join(gnn_storage_folder, 'test_prediction_confidence_dict.pkl'), 'wb') as f:
        pickle.dump(test_prediction_confidence_dict, f)

    # [5.] Save the output class for each graph in the dataset ---------------------------------------------------------
    with open(os.path.join(gnn_storage_folder, 'train_output_class_dict.pkl'), 'wb') as f:
        pickle.dump(train_output_class_dict, f)
    with open(os.path.join(gnn_storage_folder, 'test_output_class_dict.pkl'), 'wb') as f:
        pickle.dump(test_output_class_dict, f)

