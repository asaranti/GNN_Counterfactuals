"""
    Generate Kirc Dataset for testing

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-11-08
"""

import re
import os
import pickle
from pathlib import Path

from examples.synthetic_graph_examples.ba_graphs_generator import ba_graphs_gen
from preprocessing.format_transformations.format_transformation_random_kirc_to_pytorch import import_random_kirc_data

dataset_names = ["Barabasi-Albert Dataset", "Kirc Dataset"]
graph_id_composed_regex = "graph_id_[0-9]+_[0-9]+"
path = Path(__file__).parent.parent
data_folder = os.path.join(path.absolute(), "data")


def generate_data_set(dataset_name):
    """
    Generate Dataset for testing and return first patient graph.
    """
    graph_data = {}

    if dataset_name == dataset_names[0]:
        graphs_list = ba_graphs_gen(6, 10, 2, 5, 4)
    elif dataset_name == dataset_names[1]:
        dataset_pytorch_folder = os.path.join(data_folder, "output", "KIRC_RANDOM", "kirc_random_pytorch")
        with open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), 'rb') as f:
            graphs_list = pickle.load(f)

    # turn list into dictionary format
    for graph in graphs_list:
        # 2.1. Use the graph_id to "position" the graph into the "graph_adaptation_structure" ----------------------
        graph_id_composed = graph.graph_id
        pattern = re.compile(graph_id_composed_regex)
        graph_id_matches = bool(pattern.match(graph_id_composed))

        assert graph_id_matches, f"The graph's id {graph_id_composed} does not match " \
                                 f"the required pattern: {graph_id_composed_regex}"

        # 2.2. Create the initial "graph_adaptation_structure" -----------------------------------------------------
        graph_id_comp_array = graph_id_composed.split("_")
        patient_id = graph_id_comp_array[2]
        graph_id = graph_id_comp_array[3]

        # 2.3. Add dict for node_ids -------------------------------------------------------------------------------
        dict_node_ids = {}
        for i in range(0, len(graph.node_ids)):
            dict_node_ids[i] = graph.node_ids[i]
        graph.node_ids = dict_node_ids

        patient_dict = {graph_id: graph}
        graph_data[patient_id] = patient_dict

    return graph_data
