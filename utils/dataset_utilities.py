"""
    Dataset utilities

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-05-10
"""


def keep_only_first_graph_dataset(input_dataset: dict) -> list:
    """
    Since the "graph_id" counter is increased each time before a predict or retrain,
    several "versions" of the graph can be contained in the dataset. Return the dataset
    that only contains the first "version" of each graph.

    :param input_dataset: Input dataset in the form of dictionary
    :returns: Output dataset in the form of list that is going to be used for training or retraining
    """

    # [1.] Dictionary with keys the number of graph, and value the graph itself ----------------------------------------
    output_dataset_dict = {}

    for input_graph_nr_in_dataset in input_dataset:   # [2.] Iterate over all graphs -----------------------------------

        patient_graph_versions = input_dataset[input_graph_nr_in_dataset]
        patient_graph_versions_counters = [int(x) for x in list(patient_graph_versions.keys())]

        # [3.] Get the min - works also in the case where there is only one element ------------------------------------
        first_graph_cnt = min(patient_graph_versions_counters)
        output_dataset_dict[input_graph_nr_in_dataset] = patient_graph_versions[str(first_graph_cnt)]

    # [4.] Return the values as list -----------------------------------------------------------------------------------
    output_dataset = list(output_dataset_dict.values())

    return output_dataset


def keep_only_last_graph_dataset(input_dataset: dict) -> list:
    """
    Since the graph_id counter is increased each time before a predict or retrain,
    several "versions" of the graph can be contained in the dataset. Return the dataset
    that only contains the last "version" of each graph.

    :param input_dataset: Input dataset in the form of dictionary
    :returns: Output dataset in the form of list that is going to be used for training or retraining
    """

    # [1.] Dictionary with keys the number of graph, and value the graph itself ----------------------------------------
    output_dataset_dict = {}

    for input_graph_nr_in_dataset in input_dataset:  # [2.] Iterate over all graphs -----------------------------------

        patient_graph_versions = input_dataset[input_graph_nr_in_dataset]
        patient_graph_versions_counters = [int(x) for x in list(patient_graph_versions.keys())]

        # [3.] Get the min - works also in the case where there is only one element ------------------------------------
        last_graph_cnt = max(patient_graph_versions_counters)
        output_dataset_dict[input_graph_nr_in_dataset] = patient_graph_versions[str(last_graph_cnt)]

    # [4.] Return the values as list -----------------------------------------------------------------------------------
    output_dataset = list(output_dataset_dict.values())

    return output_dataset

