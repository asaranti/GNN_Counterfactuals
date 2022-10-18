"""
    GNN architecture definitions - currently the GNN architecture and size
    will be defined by the dataset that it processes. In the future
    it may be adaptive

    :author: Anna Saranti
    :copyright: © 2022 HCI-KDD (ex-AI) group
    :date: 2022-10-12
"""


def define_gnn(dataset_name: str) -> dict:
    """
    Define GNN by the dataset name

    :param dataset_name: Dataset name
    :return: Dictionary of relevant parameters of the GNN
    """

    dataset_names_list = ["kirc_random_nodes_ui", "kirc_subnet", "synthetic"]

    assert dataset_name in dataset_names_list, f"The dataset name is not admissible, " \
                                               f"it must be one of: {dataset_names_list}"

    if dataset_name == "kirc_random_nodes_ui":
        hidden_channels = 100
        layers_nr = 6
    elif dataset_name == "kirc_subnet":
        hidden_channels = 10
        layers_nr = 3
    elif dataset_name == "synthetic":
        hidden_channels = 100
        layers_nr = 5

    gnn_architecture_params_dict = {"hidden_channels": hidden_channels,
                                    "layers_nr": layers_nr}

    return gnn_architecture_params_dict
