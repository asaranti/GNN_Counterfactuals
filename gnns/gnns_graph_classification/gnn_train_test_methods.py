"""
    The GNN train() and test() functionality

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-02-18
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch_geometric.loader.dataloader import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F

from gnns.gnns_graph_classification.GCN_Graph_Classification import GCN


def train_model(model: GCN, train_loader: DataLoader, optimizer, criterion):
    """
    Training of the model

    :param model: GNN model
    :param train_loader: Loader of training set
    :param optimizer: Model optimizer
    :param criterion: Loss criterion for the GNN training
    """

    model.train()

    for data in train_loader:                               # Iterate in batches over the training dataset.

        out = model(data.x, data.edge_index, data.batch)    # Perform a single forward pass.
        loss = criterion(out, data.y)                       # Compute the loss.
        pred = out.argmax(dim=1)

        loss.backward()                                     # Derive gradients.
        optimizer.step()                                    # Update parameters based on gradients.
        optimizer.zero_grad()                               # Clear gradients.


def use_trained_model(model: GCN, data_loader: DataLoader) -> dict:
    """
    Test the model - pass all data in the test_loader to the model
    to check its performance

    :param model: GNN model
    :param data_loader: Loader of data set

    :return: Dictionary with all the gathered metrics for the test set
    """

    # [1.] Pass the test data in the GNN -------------------------------------------------------------------------------
    model.eval()

    y_test = []
    y_pred = []

    prediction_confidences_all = []
    output_classes_all = []

    for data in data_loader:                                    # Iterate in batches over the training/test dataset.

        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)                                # Use the class with highest probability.

        prediction_confidences_all.append(out)
        output_classes_all.append(pred)

        y_test += list(data.y.cpu().detach().numpy())
        y_pred += list(pred.cpu().detach().numpy())

    # [2.] Gather the predictions --------------------------------------------------------------------------------------
    prediction_confidences_all_tensor = np.around(torch.cat(prediction_confidences_all, dim=0).cpu().detach().numpy(),
                                                  decimals=2).astype('str')
    output_classes_all_tensor = torch.cat(output_classes_all, dim=0).cpu().detach().numpy().astype('str')

    # [3.] Gather the metrics ------------------------------------------------------------------------------------------
    conf_matrix = confusion_matrix(y_test, y_pred)
    true_negatives = conf_matrix[0][0]
    false_positives = conf_matrix[0][1]
    false_negatives = conf_matrix[1][0]
    true_positives = conf_matrix[1][1]

    sensitivity = true_positives/(true_positives + false_negatives)
    specificity = true_negatives/(true_negatives + false_positives)

    accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)

    dataset_metrics_dict = {"accuracy": str(round(accuracy, 2)),
                            "true_negatives": str(true_negatives),
                            "false_positives": str(false_positives),
                            "false_negatives": str(false_negatives),
                            "true_positives": str(true_positives),
                            "sensitivity": str(round(sensitivity, 2)),
                            "specificity": str(round(specificity, 2))}

    dataset_outputs_predictions_dict = {"prediction_confidences": prediction_confidences_all_tensor,
                                        "output_classes": output_classes_all_tensor}

    return dataset_metrics_dict, dataset_outputs_predictions_dict
