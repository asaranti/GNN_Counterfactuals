"""
    Individual GNNs for each of the datasets

    :author: Anna Saranti
    :copyright: © 2022 HCI-KDD (ex-AI) group
    :date: 2022-10-12
"""

from datetime import datetime
import os
import pickle
import random

from actionable.gnn_actions import GNN_Actions
from actionable.gnn_explanations import explain_sample
from actionable.graph_actions import remove_node
from gnns.gnn_selectors.gnn_definitions import define_gnn
from utils.gnn_load_save import load_gnn_model
from utils.dataset_load_save import load_action_dataset_history

########################################################################################################################
# [0.] Preparatory actions =============================================================================================
########################################################################################################################
# Set the use of GPU ---------------------------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

# Global variable containing the dictionaries --------------------------------------------------------------------------
global_gnn_models_dict = {}

# Define user ----------------------------------------------------------------------------------------------------------
user_token = "user_B"

start_session_date_time = datetime.utcnow()
start_session_date_time_str = start_session_date_time.strftime("%Y%m%d_%H%M%S")

########################################################################################################################
# [1.] Select dataset ==================================================================================================
########################################################################################################################
"""
# [1.a.] KIRC Subnet ---------------------------------------------------------------------------------------------------
dataset_name = "kirc_subnet"
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, f'{dataset_name}_pytorch.pkl'), "rb"))
gnn_architecture_params_dict = define_gnn(dataset_name)
gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, dataset_name)
# model, performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)
model = load_gnn_model(dataset_name, True, user_token)["model"]
global_gnn_models_dict['0'] = {model}

# [1.b.] KIRC random nodes ui ------------------------------------------------------------------------------------------
dataset_name = "kirc_random_nodes_ui"
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, f'{dataset_name}_pytorch.pkl'), "rb"))
gnn_architecture_params_dict = define_gnn(dataset_name)
gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, dataset_name)
model, performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)
# model = load_gnn_model(dataset_name, True, user_token)["model"]
# global_gnn_models_dict['0'] = {model}

"""
# [1.c.] Synthetic -----------------------------------------------------------------------------------------------------
dataset_name = "synthetic"
dataset_pytorch_folder = os.path.join("data", "output", "Synthetic", "synthetic_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, f'{dataset_name}_pytorch.pkl'), "rb"))
gnn_architecture_params_dict = define_gnn(dataset_name)
gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, dataset_name)
# model, performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)
model = load_gnn_model(dataset_name, False, user_token)["model"]
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
# [3.] Make some action and make a predict with the stored model =======================================================
########################################################################################################################
dataset_len = len(dataset)
graph_idx = random.randint(0, dataset_len - 1)
input_graph = dataset[graph_idx]
input_graph.to(device)
print(input_graph)

nodes_orig_nr = input_graph.x.shape[0]
print(f"Nr. of nodes original: {nodes_orig_nr}")

node_idx = 0
output_graph = remove_node(input_graph, node_idx, dataset_name, user_token)
nodes_output_nr = output_graph.x.shape[0]
print(f"Nr. of nodes after node delete: {nodes_output_nr}")

dataset[graph_idx] = output_graph

prediction_label_of_testing, prediction_confidence_of_testing = gnn_actions_obj.gnn_predict(model,
                                                                                            output_graph,
                                                                                            user_token)
print(prediction_label_of_testing, prediction_confidence_of_testing)

########################################################################################################################
# [4.] Explanation =====================================================================================================
########################################################################################################################
explanation_method = 'gnnexplainer'     # Also possible: 'ig' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ground_truth_label = int(input_graph.y.cpu().detach().numpy()[0])
explanation_label = ground_truth_label  # Can also be the opposite - all possible combinations of 0 and 1 ~~~~~~~~~~~~~~

# GNNECPLAINER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
node_mask = explain_sample(explanation_method, model, input_graph, explanation_label, user_token)
print(f"\nGNNExplainer mask: {node_mask}")

# CAPTUM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
rel_pos = list(explain_sample(
        explanation_method,
        model,
        input_graph,
        explanation_label,
        user_token
    ))
rel_pos = [str(round(edge_relevance, 2)) for edge_relevance in rel_pos]

print(rel_pos)
print(f"Captum relevances: {rel_pos}")
print(type(rel_pos[0]))

########################################################################################################################
# [5.] Retrain and store in the "global_gnn_models_dict" ===============================================================
########################################################################################################################
model, performance_values_dict = gnn_actions_obj.gnn_retrain(model, dataset, user_token)
print(performance_values_dict)

model_numbering_keys_str_list = list(global_gnn_models_dict.keys())
model_numbering_keys_int_list = [int(model_nr) for model_nr in model_numbering_keys_str_list]
max_model_nr = max(model_numbering_keys_int_list)
global_gnn_models_dict[str(max_model_nr + 1)] = model
print(global_gnn_models_dict)

########################################################################################################################
# [6.] Save the file with the start and end datetime of the session ====================================================
########################################################################################################################
end_session_date_time = datetime.utcnow()
end_session_date_time_str = end_session_date_time.strftime("%Y%m%d_%H%M%S")

dataset_user_storage_folder = os.path.join("history", "datasets", dataset_name, user_token)
dataset_history_name = f"{dataset_name}_{user_token}.txt"
dataset_history_start_end_date_time = f"{dataset_name}_{user_token}_" \
                                      f"{start_session_date_time_str}_{end_session_date_time_str}.txt"
os.rename(os.path.join(dataset_user_storage_folder, dataset_history_name),
          os.path.join(dataset_user_storage_folder, dataset_history_start_end_date_time))

########################################################################################################################
# [7.] Tryout a load of the file and execution of the actions ==========================================================
########################################################################################################################
print("===============================================================================================================")
load_action_dataset_history(dataset_name, user_token, dataset_history_start_end_date_time)
