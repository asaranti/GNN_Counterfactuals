"""
    Graph dataset save and restore

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-10-18
"""

import os


def append_action_dataset_history(dataset_name: str,
                                  user_token: str,
                                  action_description: str):
    """
    Append action in the text file that stores the user's actions.

    :param dataset_name: Dataset name that specifies the name of the model too
    :param user_token: The user token that defines the sub-folder where the actions will be stored to
    :param action_description: String that describes the action(s) that were executed
    """

    # [0.] Create/use the folder/files of storing the graph dataset information ----------------------------------------
    dataset_storage_folder = os.path.join("history", "datasets", dataset_name)
    if not os.path.exists(dataset_storage_folder):
        os.mkdir(dataset_storage_folder)
    dataset_user_storage_folder = os.path.join(dataset_storage_folder, user_token)
    if not os.path.exists(dataset_user_storage_folder):
        os.mkdir(dataset_user_storage_folder)

    dataset_history_name = f"{dataset_name}_{user_token}.txt"

    # [1.] Append the string and close the file. The newlines are presupposed to be inside the string already ----------
    dataset_history_file = open(os.path.join(dataset_user_storage_folder, dataset_history_name), "a")    # append mode
    dataset_history_file.write(action_description)
    dataset_history_file.close()


def load_action_dataset_history(dataset_name: str,
                                user_token: str):
    """
    Append action in the text file that stores the user's actions.

    :param dataset_name: Dataset name that specifies the name of the model too
    :param user_token: The user token that defines the sub-folder where the actions will be stored to
    """

    # [0.] Use the folder/files of storing the graph dataset information -----------------------------------------------
    dataset_user_storage_folder = os.path.join("history", "datasets", dataset_name, user_token)
    dataset_history_name = f"{dataset_name}_{user_token}.txt"

    # [1.] Read line-by-line the file and perform the action w.r.t. the original dataset -------------------------------
    with open(os.path.join(dataset_user_storage_folder, dataset_history_name)) as file:
        for line in file:
            print(line.rstrip())


