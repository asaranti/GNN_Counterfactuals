"""
    Test the functionality of one user performing multiple actions on different graphs.
    During those actions, all necessary data have to be stored for reconstruction and reproducibility.
    Every graph on the end dataset has to be the same as the one before the reconstruction.

    :author: Anna Saranti
    :copyright: © 2022 HCI-KDD (ex-AI) group
    :date: 2022-11-04
"""

import copy
from datetime import datetime
from itertools import combinations
import os
import pickle
import random
import string
import sys
import uuid

import numpy as np
import torch

from actionable.graph_actions import add_node, remove_node, add_edge, remove_edge, \
    add_feature_all_nodes, remove_feature_all_nodes, add_feature_all_edges, remove_feature_all_edges
from actionable.gnn_actions import GNN_Actions
from gnns.gnn_selectors.gnn_definitions import define_gnn
from utils.dataset_load import load_action_dataset_history
from utils.gnn_load_save import load_gnn_model
from utils.graph_utilities import graphs_equal

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

"""
# [A.a.] KIRC Subnet ---------------------------------------------------------------------------------------------------
dataset_name = "kirc_subnet"
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, f'{dataset_name}_pytorch.pkl'), "rb"))
gnn_architecture_params_dict = define_gnn(dataset_name)
gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, dataset_name)
# model, performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)
already_trained_model_dict = load_gnn_model(dataset_name, True, user_token)
model = already_trained_model_dict["model"]
global_gnn_models_dict['0'] = {model}
"""

# [A.b.] KIRC random nodes ui ------------------------------------------------------------------------------------------
dataset_name = "kirc_random_nodes_ui"
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, f'{dataset_name}_pytorch.pkl'), "rb"))
gnn_architecture_params_dict = define_gnn(dataset_name)
gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, dataset_name)
# model, performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)
already_trained_model_dict = load_gnn_model(dataset_name, False, user_token)
model = already_trained_model_dict["model"]
# global_gnn_models_dict['0'] = {model}

"""
# [A.c.] Synthetic -----------------------------------------------------------------------------------------------------
dataset_name = "synthetic"
dataset_pytorch_folder = os.path.join("data", "output", "Synthetic", "synthetic_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, f'{dataset_name}_pytorch.pkl'), "rb"))
gnn_architecture_params_dict = define_gnn(dataset_name)
gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, dataset_name)
# model, performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)
already_trained_model_dict = load_gnn_model(dataset_name, True, user_token)
model = already_trained_model_dict["model"]
"""

train_set_metrics_dict = already_trained_model_dict["train_set_metrics_dict"]
test_set_metrics_dict = already_trained_model_dict["test_set_metrics_dict"]
print(f"Train set performance: {train_set_metrics_dict}")
print(f"Test set performance: {test_set_metrics_dict}")
print("---------------------------------------------------------------------------------------------")
global_gnn_models_dict['0'] = {model}

########################################################################################################################
# [B.] Create the folder that is going to keep the history of the interactions of this user with this dataset ----------
########################################################################################################################
dataset_storage_folder = os.path.join("history", "datasets", dataset_name)
if not os.path.exists(dataset_storage_folder):
    os.mkdir(dataset_storage_folder)
dataset_user_storage_folder = os.path.join(dataset_storage_folder, user_token)
if not os.path.exists(dataset_user_storage_folder):
    os.mkdir(dataset_user_storage_folder)

########################################################################################################################
# [C.] Select randomly some actions on the same graph ==================================================================
########################################################################################################################
dataset_len = len(dataset)

min_nr_graphs_changed = 20
max_nr_graphs_changed = 50

graphs_to_change_nr = random.randint(min_nr_graphs_changed, max_nr_graphs_changed)

for graph_to_change_idx in range(graphs_to_change_nr):

    graph_idx = random.randint(0, dataset_len - 1)
    print(f"Graph_idx: {graph_idx}")
    input_graph = dataset[graph_idx]

    possible_graph_actions_nr = 8
    min_nr_actions = 20
    max_nr_actions = 50
    actions_of_change_nr = random.randint(min_nr_actions, max_nr_actions)
    letters = string.ascii_lowercase

    for action_of_change_idx in range(actions_of_change_nr):

        actions_of_change_nr = random.randint(1, possible_graph_actions_nr)

        ################################################################################################################
        if actions_of_change_nr == 1:       # [1.] add_node >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            input_features_graph_nr = input_graph.x.cpu().detach().numpy().shape[1]
            node_features = np.random.rand(1, input_features_graph_nr).astype(np.float32)
            node_label = "node_label_" + str(uuid.uuid4())
            node_id = "node_id_" + str(uuid.uuid4())

            output_graph = add_node(
                input_graph,
                node_features,
                node_label,
                node_id,
                dataset_name,
                user_token,
                True
                )

        ################################################################################################################
        elif actions_of_change_nr == 2:     # [2.] remove_node >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            graph_nodes_nr = input_graph.num_nodes
            if graph_nodes_nr <= 1:

                output_graph = input_graph
                print("There are no nodes left to remove.")
            else:

                node_index_to_remove = random.randint(0, graph_nodes_nr - 1)

                output_graph = remove_node(
                    input_graph,
                    node_index_to_remove,
                    dataset_name,
                    user_token,
                    True
                )

        ################################################################################################################
        elif actions_of_change_nr == 3:     # [3.] add_edge >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            # [3.1.] Get all "edge_index" ------------------------------------------------------------------------------
            input_graph_edge_index = input_graph.edge_index.cpu().detach().numpy()
            if input_graph_edge_index is not None:
                input_graph_edge_index_left = list(input_graph_edge_index[0, :])
                input_graph_edge_index_right = list(input_graph_edge_index[1, :])
            else:
                input_graph_edge_index_left = []
                input_graph_edge_index_right = []

            input_graph_edge_index_list = [tuple(sorted(tuple(row))) for row in input_graph_edge_index.T]
            input_graph_edge_index_set = set(input_graph_edge_index_list)

            # [3.2.] Add a new edge where there is no one yet ----------------------------------------------------------
            #        The only possibility that adding an edge is not possible is the graph being complete --------------
            graph_nodes_nr = input_graph.num_nodes
            all_edge_combinations_list = list(combinations(list(range(graph_nodes_nr)), 2))
            all_edge_combinations_set = set(all_edge_combinations_list)

            # [3.3.] Compute the diff between all node pairs (all possible edges) --------------------------------------
            #        and pick one of those randomly - i.e. create the edge ---------------------------------------------
            edges_remaining_set = all_edge_combinations_set - input_graph_edge_index_set
            edges_remaining_list = list(edges_remaining_set)

            # [3.4.] Check that the subtraction of the sets returns the expected results; the one must be the subset ---
            #        of the other. If the pairs of egdes aren't sorted, then this equality does not hold. --------------
            all_edge_combinations_list_len = len(all_edge_combinations_list)
            input_graph_edge_index_list_len = len(input_graph_edge_index_list)
            edges_remaining_list_len = len(edges_remaining_list)

            assert all_edge_combinations_list_len - input_graph_edge_index_list_len == edges_remaining_list_len, \
                f"The length of all possible edges: {all_edge_combinations_list_len} -minus- the length of the " \
                f"already existing edges: {input_graph_edge_index_list_len} has to be equal to the length of the " \
                f"list of the remaining edges: {edges_remaining_list_len}."

            if edges_remaining_list_len == 0:
                print("The graph is fully connected, a new edge cannot be generated.")
            else:

                # [3.5.] Pick one of the edges that you can lay --------------------------------------------------------
                new_edge_remaining_idx = random.randint(0, edges_remaining_list_len - 1)
                new_edge = edges_remaining_list[new_edge_remaining_idx]
                new_edge_index_left = new_edge[0]
                new_edge_index_right = new_edge[1]

                # [3.6.] If there are already edge attributes, then generate some artifical values for them ------------
                new_edge_attr = None
                if hasattr(input_graph, 'edge_attr') and input_graph.edge_attr is not None:

                    input_edge_features_nr = input_graph.edge_attr.shape[1]
                    new_edge_attr = np.random.rand(1, input_edge_features_nr).astype(np.float32)

                # [3.7.] Add the edge ----------------------------------------------------------------------------------
                output_graph = add_edge(
                    input_graph,
                    new_edge_index_left,
                    new_edge_index_right,
                    new_edge_attr,
                    dataset_name,
                    user_token,
                    True
                    )

        ################################################################################################################
        elif actions_of_change_nr == 4:     # [4.] remove_edge >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            # [4.1.] Get all "edge_index" ------------------------------------------------------------------------------
            input_graph_edge_index = input_graph.edge_index.cpu().detach().numpy()

            if input_graph_edge_index is not None:

                input_graph_edge_index_left = list(input_graph_edge_index[0, :])
                input_graph_edge_index_right = list(input_graph_edge_index[1, :])

                edges_nr = np.shape(input_graph_edge_index)[1]

                if edges_nr == 0:

                    output_graph = input_graph
                    print("There are no edges left to remove.")
                else:
                    # [4.2.] Select random edge and remove it ----------------------------------------------------------
                    selected_random_edge = random.randint(0, edges_nr-1)

                    output_graph = remove_edge(
                        input_graph,
                        input_graph_edge_index_left[selected_random_edge],
                        input_graph_edge_index_right[selected_random_edge],
                        dataset_name,
                        user_token,
                        True
                    )
            else:
                output_graph = input_graph
                print("There are no edges left to remove.")

        ################################################################################################################
        elif actions_of_change_nr == 5:     # [5.] add_feature_all_nodes >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            # Adding one column to all nodes ---------------------------------------------------------------------------
            feature_label = "feature_label_" + str(uuid.uuid4())

            input_nodes = input_graph.num_nodes
            new_input_node_feature = np.random.rand(1, input_nodes).astype(np.float32)

            output_graph = add_feature_all_nodes(
                input_graph,
                new_input_node_feature,
                feature_label,
                dataset_name,
                user_token,
                True
            )

        ################################################################################################################
        elif actions_of_change_nr == 6:     # [6.] remove_feature_all_nodes >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            input_features_graph_nr = input_graph.x.size(dim=1)
            if input_features_graph_nr <= 1:
                output_graph = input_graph
                print("The number of input features of the nodes is less than 1, a removal is not acceptable.")
            else:

                removed_node_feature_idx = random.randint(0, input_features_graph_nr-1)

                output_graph = remove_feature_all_nodes(
                    input_graph,
                    removed_node_feature_idx,
                    dataset_name,
                    user_token,
                    True
                )

        ################################################################################################################
        elif actions_of_change_nr == 7:     # [7.] add_feature_all_edges >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            edges_nr = input_graph.num_edges

            if edges_nr == 0:

                output_graph = input_graph
                print("The number of edges is 0. An edge feature cannot be added.")
            else:

                new_input_edge_feature = np.random.rand(1, edges_nr).astype(np.float32)

                feature_label = "feature_label_" + str(uuid.uuid4())

                output_graph = add_feature_all_edges(
                    input_graph,
                    new_input_edge_feature,
                    feature_label,
                    dataset_name,
                    user_token,
                    True
                )

        ################################################################################################################
        elif actions_of_change_nr == 8:     # [8.] remove_feature_all_edges >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            if input_graph.edge_attr is None:

                output_graph = input_graph
                print("The \"edge_attr\" of the graph is None, no edge feature removal is possible.")
            else:
                input_graph_edge_attr = input_graph.edge_attr.cpu().detach().numpy()
                edge_attributes_nr = input_graph_edge_attr.shape[1]

                if edge_attributes_nr <= 0:
                    output_graph = input_graph
                    print("The number of input features of the edges is less than 0, a removal is not acceptable.")
                else:
                    print(f"edge_attributes_nr: {edge_attributes_nr}")
                    removed_edge_attribute_idx = random.randint(0, edge_attributes_nr-1)

                    output_graph = remove_feature_all_edges(
                        input_graph,
                        removed_edge_attribute_idx,
                        dataset_name,
                        user_token,
                        True
                    )

        else:
            assert False, "The number of admissible actions on graphs is 8."

        ################################################################################################################
        # [9.] Update the input_graph with the previous output graph ---------------------------------------------------
        input_graph = output_graph

    changed_graph = copy.deepcopy(input_graph)
    dataset[graph_idx] = changed_graph

########################################################################################################################
# [D.] Save the file with the start and end datetime of the session ====================================================
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
# [E.] Restore the graph from the stored history files =================================================================
########################################################################################################################
print("===============================================================================================================")
print("=========== Restore the whole dataset =========================================================================")
print("===============================================================================================================")
# [1.] The loaded dataset from the history files -----------------------------------------------------------------------
loaded_dataset = load_action_dataset_history(dataset_name, user_token, dataset_history_start_end_date_time)

# [2.] Control each graph in both the datsets --------------------------------------------------------------------------
#      the changed graphs have to be changed and the rest have to stay unchanged ---------------------------------------
datasets_equal = True
for graph_idx in range(len(loaded_dataset)):

    changed_graph = dataset[graph_idx]
    loaded_graph = loaded_dataset[graph_idx]

    # [3.] Graph equality check - if false then stop -------------------------------------------------------------------
    graph_equality = graphs_equal(changed_graph, loaded_graph)

    if graph_equality is False:
        datasets_equal = False
        print(f"BUG: The changed graph on index {graph_idx} and the corresponding loaded one from the history file "
              f"are unequal.\nThe two datasets differ, no other comparison needed.\nFix your bugs.")
        break

# [4.] If you reached this point, then all graphs are equal ------------------------------------------------------------
if datasets_equal:
    print("CONGRATS! ALL graphs are equal !!!!!!!!!!!!!!!!!!!!")

