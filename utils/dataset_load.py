"""
    Graph dataset graph restore

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-10-18
"""

import os
import pickle

import numpy as np

from actionable.graph_actions import add_node, remove_node, add_edge, remove_edge, \
    add_feature_all_nodes, remove_feature_all_nodes, add_feature_all_edges, remove_feature_all_edges
from utils.dataset_utilities import check_allowable_datasets


def load_action_dataset_history(dataset_name: str,
                                user_token: str,
                                dataset_history_name: str):
    """
    Append action in the text file that stores the user's actions.

    :param dataset_name: Dataset name that specifies the name of the model too
    :param user_token: The user token that defines the sub-folder where the actions will be stored to
    :param dataset_history_name: The name of the file containing the history. It might be the currently
                                 changed one or some of an older session.
    """

    # [0.] Use the folder/files of storing the graph dataset information -----------------------------------------------
    dataset_user_storage_folder = os.path.join("history", "datasets", dataset_name, user_token)

    # [1.] Load the initial dataset ------------------------------------------------------------------------------------
    check_allowable_datasets(dataset_name)

    if dataset_name == "kirc_subnet":
        dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
        dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, f'{dataset_name}_pytorch.pkl'), "rb"))
    elif dataset_name == "kirc_random_nodes_ui":
        dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
        dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, f'{dataset_name}_pytorch.pkl'), "rb"))
    elif dataset_name == "synthetic":
        dataset_pytorch_folder = os.path.join("data", "output", "Synthetic", "synthetic_pytorch")
        dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, f'{dataset_name}_pytorch.pkl'), "rb"))

    # [2.] Read line-by-line the file and perform the action w.r.t. the original dataset -------------------------------
    with open(os.path.join(dataset_user_storage_folder, dataset_history_name)) as file:
        for line in file:
            action_str = line.rstrip()
            action_array = action_str.split(",")
            print(action_array)

            # [2.1.] First arg is the graph_id, the second one is the function name ------------------------------------
            #        and the last one is the datetime (not used) -------------------------------------------------------
            graph_id_str = action_array[0].split(":")[1]
            graph_id_array = graph_id_str.split("_")
            graph_index_in_dataset = int(graph_id_array[2])
            input_graph = dataset[graph_index_in_dataset]

            func_name = action_array[1]
            date_time = action_array[-1]

            # [2.2.] Call each of the functions separately -------------------------------------------------------------
            if func_name == "add_node":

                node_features = np.fromstring(action_array[2].split(":")[1], sep=' ')
                label = action_array[3].split(":")[1]
                node_id = action_array[4].split(":")[1]

                output_graph = add_node(input_graph=input_graph,
                                        node_features=node_features,
                                        label=label,
                                        node_id=node_id
                                        )

            elif func_name == "remove_node":

                node_index = int(action_array[2].split(":")[1])

                output_graph = remove_node(input_graph=input_graph,
                                           node_index=node_index
                                           )

            elif func_name == "add_edge":

                new_edge_index_left = int(action_array[2].split(":")[1])
                new_edge_index_right = int(action_array[3].split(":")[1])
                # TODO: new_edge_attr >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TODO >>>>>>>>>>>>>>>>>>>>>>>>

                output_graph = add_edge(input_graph=input_graph,
                                        new_edge_index_left=new_edge_index_left,
                                        new_edge_index_right=new_edge_index_right,
                                        new_edge_attr=None,
                                        )

            elif func_name == "remove_edge":

                edge_index_left = int(action_array[2].split(":")[1])
                edge_index_right = int(action_array[3].split(":")[1])

                remove_edge(input_graph=input_graph,
                            edge_index_left=edge_index_left,
                            edge_index_right=edge_index_right
                            )

            elif func_name == "add_feature_all_nodes":

                new_input_node_feature = np.fromstring(action_array[2].split(":")[1], sep=' ')

                add_feature_all_nodes(input_graph=input_graph,
                                      new_input_node_feature=new_input_node_feature,
                                      label=label
                                      )

            elif func_name == "remove_feature_all_nodes":

                removed_node_feature_idx = int(action_array[2].split(":")[1])

                remove_feature_all_nodes(input_graph=input_graph,
                                         removed_node_feature_idx=removed_node_feature_idx
                                         )

            elif func_name == "add_feature_all_edges":

                new_input_edge_feature = np.fromstring(action_array[2].split(":")[1], sep=' ')

                add_feature_all_edges(input_graph=input_graph,
                                      new_input_edge_feature=new_input_edge_feature
                                      )

            elif func_name == "remove_feature_all_edges":

                removed_edge_attribute_idx = int(action_array[2].split(":")[1])

                remove_feature_all_edges(input_graph=input_graph,
                                         removed_edge_attribute_idx=removed_edge_attribute_idx
                                         )

            else:
                assert False, f"The graph action method: {func_name} is not in graph_actions.py"

            # [3.] Update the dataset with the new graph ---------------------------------------------------------------
            dataset[graph_index_in_dataset] = output_graph

