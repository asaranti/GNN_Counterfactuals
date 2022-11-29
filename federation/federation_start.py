"""
    Federation Example

    :author: Anna Saranti
    :copyright: © 2022 HCI-KDD (ex-AI) group
    :date: 2022-11-29
"""

from datetime import datetime
import os
import pickle
import sys

import numpy as np
import torch

from actionable.gnn_actions import GNN_Actions
from gnns.gnn_selectors.gnn_definitions import define_gnn
from utils.gnn_load_save import load_gnn_model

np.set_printoptions(threshold=sys.maxsize)

########################################################################################################################
# [0.] Preparatory actions =============================================================================================
########################################################################################################################
# Set the use of GPU ---------------------------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)

# Global variable containing the dictionaries --------------------------------------------------------------------------
global_gnn_models_dict = {}

# Define user ----------------------------------------------------------------------------------------------------------
user_token = "user_C"

start_session_date_time = datetime.utcnow()
start_session_date_time_str = start_session_date_time.strftime("%Y%m%d_%H%M%S")

########################################################################################################################
# [A.] Select dataset ==================================================================================================
########################################################################################################################
# [A.c.] Synthetic -----------------------------------------------------------------------------------------------------
dataset_name = "synthetic"
dataset_pytorch_folder = os.path.join("data", "output", "Synthetic", "synthetic_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, f'{dataset_name}_pytorch.pkl'), "rb"))
gnn_architecture_params_dict = define_gnn(dataset_name)
gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, dataset_name)
# model, performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)
already_trained_model_dict = load_gnn_model(dataset_name, True, user_token)
model = already_trained_model_dict["model"]

train_set_metrics_dict = already_trained_model_dict["train_set_metrics_dict"]
test_set_metrics_dict = already_trained_model_dict["test_set_metrics_dict"]
print(f"Train set performance: {train_set_metrics_dict}")
print(f"Test set performance: {test_set_metrics_dict}")
print("---------------------------------------------------------------------------------------------")
global_gnn_models_dict['0'] = {model}

########################################################################################################################
# [B.] Split the training set ==========================================================================================
########################################################################################################################
clients_nr = 2

print(f"Training set shuffled indexes: {gnn_actions_obj.train_dataset_shuffled_indexes}")

#
