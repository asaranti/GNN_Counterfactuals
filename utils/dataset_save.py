"""
    Graph dataset actions save (append)

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-10-31
"""

from datetime import datetime

import os


def check_dataset_history_file(dataset_name: str,
                               user_token: str):
    """
    Check that the dataset history file that is being continuously written did not remain there "unclosed"
    from a previous abruptly closed session - because of some exception or any other stop.
    Use the current datetime to rename it with that only for backup reasons and inform the user.

    :param dataset_name: Dataset name that specifies the name of the model too
    :param user_token: The user token that defines the sub-folder where the actions will be stored to
    """

    # [0.] Specify the file for storing the graph dataset information --------------------------------------------------
    #      The folder and the file need to be created already ----------------------------------------------------------
    dataset_user_storage_folder = os.path.join("history", "datasets", dataset_name, user_token)
    dataset_history_name = f"{dataset_name}_{user_token}.txt"
    dataset_history_file = os.path.join(dataset_user_storage_folder, dataset_history_name)

    # [1.] If the file already exists, that means that some previous session was not closed properly -------------------
    b_dataset_history_file = os.path.exists(dataset_history_file)

    if b_dataset_history_file:

        session_date_time = datetime.utcnow()
        session_date_time_str = session_date_time.strftime("%Y%m%d_%H%M%S")

        backup_unclosed_history_file = f"{dataset_name}_{user_token}_{session_date_time_str}.txt"

        print(f"\nThe dataset history file {dataset_history_name} already exists\n. "
              f"It seems that the previous session was closed abruptly.\n"
              f"It will be renamed to: {backup_unclosed_history_file}.\n\n")

        os.rename(dataset_history_file,
                  os.path.join(dataset_user_storage_folder, backup_unclosed_history_file))


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


