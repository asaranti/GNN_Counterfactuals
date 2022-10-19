"""
    GNN model save and restore

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-10-13
"""

from datetime import datetime
import os
import pickle
import shutil

import torch

from gnns.gnns_graph_classification.GCN_Graph_Classification import GCN


def load_gnn_model(dataset_name: str, b_initial: bool, user_token: str = None) -> dict:
    """
    Load all model-relevant information from local storage

    [1.] The model in ".pth" format
    [2.] The overall performance of the model (confusion matrix, sensitivity, specificity)
    [3.] Two lists of indexes of the graph ids for the training and the test set
    [4.] The exact prediction confidence for each graph in the dataset
    [5.] The predicted class for each graph in the dataset

    :param dataset_name: Dataset name that specifies the name of the model too
    :param b_initial: Load the initial or the latest model
    :param user_token: The user token that defines the sub-folder where the model will be loaded from
                       If b_initial is True, then any "user_token" content will be ignored

    :return: Dictionary of all the restored data
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = 'cuda:0'

    if b_initial and user_token is not None:
        print("\"b_initial\" is True, the content of \"user_token\" will be ignored")

    gnn_storage_folder = os.path.join("history", "models", dataset_name)

    # [0.] If init get the initial, else the latest one after the last retrain - if exists one -------------------------
    if b_initial:
        gnn_storage_subfolder = os.path.join(gnn_storage_folder, "init_" + dataset_name)
    else:
        # Use the latest one after the last retrain - if there was one. ------------------------------------------------
        # If there wasn't one before, then use the one of init ---------------------------------------------------------
        if os.path.exists(os.path.join(gnn_storage_folder, user_token)):
            gnn_storage_subfolder = os.path.join(gnn_storage_folder, user_token, "latest_" + dataset_name)
        else:
            gnn_storage_subfolder = os.path.join(gnn_storage_folder, "init_" + dataset_name)

    # [1.] The model in ".pth" format ----------------------------------------------------------------------------------
    model = torch.load(os.path.join(gnn_storage_subfolder, f"{dataset_name}_model.pth")).to(device)
    model.eval()

    # [2.] The overall performance of the model (confusion matrix, sensitivity, specificity) ---------------------------
    train_set_metrics_dict = pickle.load(open(os.path.join(gnn_storage_subfolder, 'train_set_metrics_dict.pkl'), "rb"))
    test_set_metrics_dict = pickle.load(open(os.path.join(gnn_storage_subfolder, 'test_set_metrics_dict.pkl'), "rb"))

    # [3.] Two lists of indexes of the graph ids for the training and the test set -------------------------------------
    train_dataset_shuffled_indexes = pickle.load(open(os.path.join(gnn_storage_subfolder,
                                                                   'train_dataset_shuffled_indexes.pkl'), "rb"))
    test_dataset_shuffled_indexes = pickle.load(open(os.path.join(gnn_storage_subfolder,
                                                                  'test_dataset_shuffled_indexes.pkl'), "rb"))

    # [4.] Load the exact prediction confidence for each graph in the dataset ------------------------------------------
    train_prediction_confidence_dict = pickle.load(open(os.path.join(gnn_storage_subfolder,
                                                   'train_prediction_confidence_dict.pkl'), "rb"))
    test_prediction_confidence_dict = pickle.load(open(os.path.join(gnn_storage_subfolder,
                                                  'test_prediction_confidence_dict.pkl'), "rb"))

    # [5.] Load the output class for each graph in the dataset  --------------------------------------------------------
    train_output_class_dict = pickle.load(open(os.path.join(gnn_storage_subfolder,
                                          'train_output_class_dict.pkl'), "rb"))
    test_output_class_dict = pickle.load(open(os.path.join(gnn_storage_subfolder,
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
                   dataset_name: str,
                   b_initial: bool,
                   user_token: str = None):
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
    :param b_initial: Is it the initial save or another which comes at retrain
    :param user_token: The user token that defines the sub-folder where the model will be stored to
                       If b_initial is True, then any "user_token" content will be ignored
    """

    if b_initial and user_token is not None:
        print("\"b_initial\" is True, the content of \"user_token\" will be ignored")

    # [0.1.] Create the folder of storing the GNN-relevant information -------------------------------------------------
    gnn_storage_folder = os.path.join("history", "models", dataset_name)
    if not os.path.exists(gnn_storage_folder):
        os.makedirs(gnn_storage_folder)

    if not b_initial:
        gnn_storage_folder = os.path.join(gnn_storage_folder, user_token)
        if not os.path.exists(gnn_storage_folder):
            os.makedirs(gnn_storage_folder)

    # [0.2.] If it is the initial save of the model information, then it is saved in a subfolder -----------------------
    #        starting with the "init_" substring -----------------------------------------------------------------------
    gnn_storage_subfolder = None
    if b_initial:
        init_gnn_storage_folder = os.path.join(gnn_storage_folder, "init_" + dataset_name)
        assert not os.path.exists(init_gnn_storage_folder), \
            f"The initial model for the dataset {dataset_name} is already created, a rewrite is not possible"
        os.makedirs(init_gnn_storage_folder)
        gnn_storage_subfolder = init_gnn_storage_folder
    else:
        date_time = datetime.utcnow()
        str_date_time = date_time.strftime("%Y%m%d_%H%M%S")
        current_gnn_storage_folder = os.path.join(gnn_storage_folder, f"{str_date_time}_" + dataset_name)
        os.makedirs(current_gnn_storage_folder)
        gnn_storage_subfolder = current_gnn_storage_folder

    # [1.] Save the model in ".pth" format -----------------------------------------------------------------------------
    gnn_model_file_path = os.path.join(gnn_storage_subfolder, f"{dataset_name}_model.pth")
    torch.save(model, gnn_model_file_path)

    # [2.] Save the overall performance of the model (confusion matrix, sensitivity, specificity) ----------------------
    #      for the training set and the test set [pickle] --------------------------------------------------------------
    with open(os.path.join(gnn_storage_subfolder, 'train_set_metrics_dict.pkl'), 'wb') as f:
        pickle.dump(train_set_metrics_dict, f)
    with open(os.path.join(gnn_storage_subfolder, 'test_set_metrics_dict.pkl'), 'wb') as f:
        pickle.dump(test_set_metrics_dict, f)

    # [3.] Save the two lists of indexes of the graph ids for the training and the test set ----------------------------
    with open(os.path.join(gnn_storage_subfolder, 'train_dataset_shuffled_indexes.pkl'), 'wb') as f:
        pickle.dump(train_dataset_shuffled_indexes, f)
    with open(os.path.join(gnn_storage_subfolder, 'test_dataset_shuffled_indexes.pkl'), 'wb') as f:
        pickle.dump(test_dataset_shuffled_indexes, f)

    # [4.] Save the exact prediction confidence for each graph in the dataset ------------------------------------------
    with open(os.path.join(gnn_storage_subfolder, 'train_prediction_confidence_dict.pkl'), 'wb') as f:
        pickle.dump(train_prediction_confidence_dict, f)
    with open(os.path.join(gnn_storage_subfolder, 'test_prediction_confidence_dict.pkl'), 'wb') as f:
        pickle.dump(test_prediction_confidence_dict, f)

    # [5.] Save the output class for each graph in the dataset ---------------------------------------------------------
    with open(os.path.join(gnn_storage_subfolder, 'train_output_class_dict.pkl'), 'wb') as f:
        pickle.dump(train_output_class_dict, f)
    with open(os.path.join(gnn_storage_subfolder, 'test_output_class_dict.pkl'), 'wb') as f:
        pickle.dump(test_output_class_dict, f)

    # [6.] Write and keep the "latest_" folder -------------------------------------------------------------------------
    if not b_initial:
        latest_gnn_storage_subfolder = os.path.join(gnn_storage_folder, "latest_" + dataset_name)
        if os.path.exists(latest_gnn_storage_subfolder):
            shutil.rmtree(latest_gnn_storage_subfolder)
        shutil.copytree(gnn_storage_subfolder, latest_gnn_storage_subfolder)
