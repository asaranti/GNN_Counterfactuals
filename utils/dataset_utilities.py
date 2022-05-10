"""
    Dataset utilities

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-05-10
"""

import re


def keep_only_last_graph_dataset(input_dataset: list) -> list:
    """
    Since the graph_id counter is increased each time before a predict or retrain,
    several "versions" of the graph can be contained in the dataset. Return the dataset
    that only contains the last "version" of each graph.

    :param input_dataset: Input dataset
    :returns: Output dataset
    """

    # [1.] Dictionary with keys the number of graph, and value the graph itself ----------------------------------------
    output_dataset_dict = {}

    for input_graph in input_dataset:   # Iterate over all graphs ------------------------------------------------------

        input_graph_numbering_ids = re.findall('[0-9]+', input_graph.graph_id)
        input_graph_nr_in_dataset = input_graph_numbering_ids[0]
        input_graph_cnt_in_dataset = input_graph_numbering_ids[1]

        # [2.] If the graph number is already in the keys of the dictionary, then check the counter --------------------
        if input_graph_nr_in_dataset in output_dataset_dict:
            graph_already_in_dict = output_dataset_dict[input_graph_nr_in_dataset]
            already_graph_cnt_in_dataset = re.findall('[0-9]+', graph_already_in_dict.graph_id)[1]
            if already_graph_cnt_in_dataset < input_graph_cnt_in_dataset:   # The current counter is higher than the
                # present one in the dictionary - update the graph -----------------------------------------------------

                output_dataset_dict[input_graph_nr_in_dataset] = input_graph
        # [3.] First time that one sees the graph with this number -----------------------------------------------------
        else:
            output_dataset_dict[input_graph_nr_in_dataset] = input_graph

    # [4.] Return the values as list -----------------------------------------------------------------------------------
    output_dataset = list(output_dataset_dict.values())

    return output_dataset

