"""
    GNN utilities - Save and restore the model

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-10-13
"""

import os

import torch

from gnns.gnns_graph_classification.GCN_Graph_Classification import GCN


def save_gnn_model(model: GCN, dataset_name: str):
    """
    Save the model locally

    :param model: Model to be saved
    :param dataset_name: Dataset name that specifies the name of the model too
    """

    gnn_storage_folder = "models"
    gnn_model_file_path = os.path.join(gnn_storage_folder, f"{dataset_name}_model.pth")
    torch.save(model, gnn_model_file_path)
