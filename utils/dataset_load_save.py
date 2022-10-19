"""
    Graph dataset save and restore

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-10-18
"""

import os
import pickle

from utils.dataset_utilities import check_allowable_datasets


def append_action_dataset_history(dataset_name: str,
                                  user_token: str,
                                  action_description: str):
    """
    Append action in the text file that stores the user's actions.

    :param dataset_name: Dataset name that specifies the name of the model too
    :param user_token: The user token that defines the sub-folder where the actions will be stored to
    :param action_description: String that describes the action(s) that were executed
    """

    # [0.] Specify the file for storing the graph dataset information --------------------------------------------------
    #      The folder and the file need to be created already ----------------------------------------------------------
    dataset_user_storage_folder = os.path.join("history", "datasets", dataset_name, user_token)
    dataset_history_name = f"{dataset_name}_{user_token}.txt"

    # [1.] Append the string and close the file. The newlines are presupposed to be inside the string already ----------
    dataset_history_file = open(os.path.join(dataset_user_storage_folder, dataset_history_name), "a")    # append mode
    dataset_history_file.write(action_description)
    dataset_history_file.close()


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
