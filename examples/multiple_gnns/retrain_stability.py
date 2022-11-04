"""
    Excercise the retrain stability

    :author: Anna Saranti
    :copyright: © 2022 HCI-KDD (ex-AI) group
    :date: 2022-11-04
"""

from datetime import datetime
import os
import pickle

from actionable.gnn_actions import GNN_Actions
from gnns.gnn_selectors.gnn_definitions import define_gnn
from utils.gnn_load_save import load_gnn_model

########################################################################################################################
# [0.] Preparatory actions =============================================================================================
########################################################################################################################
# Set the use of GPU ---------------------------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

# Global variable containing the dictionaries --------------------------------------------------------------------------
global_gnn_models_dict = {}

# Define user ----------------------------------------------------------------------------------------------------------
user_token = "user_C"

start_session_date_time = datetime.utcnow()
start_session_date_time_str = start_session_date_time.strftime("%Y%m%d_%H%M%S")

########################################################################################################################
# [1.] Select dataset ==================================================================================================
########################################################################################################################

# [1.a.] KIRC Subnet ---------------------------------------------------------------------------------------------------
dataset_name = "kirc_subnet"
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, f'{dataset_name}_pytorch.pkl'), "rb"))
gnn_architecture_params_dict = define_gnn(dataset_name)
gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, dataset_name)
# model, performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)
already_trained_model_dict = load_gnn_model(dataset_name, False, user_token)
model = already_trained_model_dict["model"]
global_gnn_models_dict['0'] = {model}

"""
# [1.b.] KIRC random nodes ui ------------------------------------------------------------------------------------------
dataset_name = "kirc_random_nodes_ui"
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, f'{dataset_name}_pytorch.pkl'), "rb"))
gnn_architecture_params_dict = define_gnn(dataset_name)
gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, dataset_name)
# model, performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)
already_trained_model_dict = load_gnn_model(dataset_name, False, user_token)
model = already_trained_model_dict["model"]
global_gnn_models_dict['0'] = {model}
"""

"""
# [1.c.] Synthetic -----------------------------------------------------------------------------------------------------
dataset_name = "synthetic"
dataset_pytorch_folder = os.path.join("data", "output", "Synthetic", "synthetic_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, f'{dataset_name}_pytorch.pkl'), "rb"))
gnn_architecture_params_dict = define_gnn(dataset_name)
gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, dataset_name)
# model, performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)
already_trained_model_dict = load_gnn_model(dataset_name, False, user_token)
model = already_trained_model_dict["model"]
"""

train_set_metrics_dict = already_trained_model_dict["train_set_metrics_dict"]
test_set_metrics_dict = already_trained_model_dict["test_set_metrics_dict"]
print(f"Train set performance: {train_set_metrics_dict}")
print(f"Test set performance: {test_set_metrics_dict}")
print("---------------------------------------------------------------------------------------------")
global_gnn_models_dict['0'] = {model}

########################################################################################################################
# [2.] Create the folder that is going to keep the history of the interactions of this user with this dataset ----------
########################################################################################################################
dataset_storage_folder = os.path.join("history", "datasets", dataset_name)
if not os.path.exists(dataset_storage_folder):
    os.mkdir(dataset_storage_folder)
dataset_user_storage_folder = os.path.join(dataset_storage_folder, user_token)
if not os.path.exists(dataset_user_storage_folder):
    os.mkdir(dataset_user_storage_folder)

########################################################################################################################
# [3.] Retrain and store in the "global_gnn_models_dict" ===============================================================
########################################################################################################################
retrains_nr = 10
for retrain_idx in range(retrains_nr):

    print(f"Retrain idx: {retrain_idx}")

    model, performance_values_dict = gnn_actions_obj.gnn_retrain(model, dataset, user_token)
    print("---------------------------------------------------------------------------------------------")
