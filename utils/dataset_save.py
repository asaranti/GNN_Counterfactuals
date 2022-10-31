"""
    Graph dataset actions save (append)

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-10-31
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

    # [0.] Specify the file for storing the graph dataset information --------------------------------------------------
    #      The folder and the file need to be created already ----------------------------------------------------------
    dataset_user_storage_folder = os.path.join("history", "datasets", dataset_name, user_token)
    dataset_history_name = f"{dataset_name}_{user_token}.txt"

    # [1.] Append the string and close the file. The newlines are presupposed to be inside the string already ----------
    dataset_history_file = open(os.path.join(dataset_user_storage_folder, dataset_history_name), "a")    # append mode
    dataset_history_file.write(action_description)
    dataset_history_file.close()
