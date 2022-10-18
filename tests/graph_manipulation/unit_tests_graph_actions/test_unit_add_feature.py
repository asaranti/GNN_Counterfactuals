"""
    Test add feature(s)

    :author: David Kerschbaumer
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-05-26
"""

import os
import pickle
import random
import uuid

import numpy as np
import torch_geometric

from actionable.graph_actions import add_feature_all_nodes
from tests.utils_tests.utils_tests_graph_actions.utilities_for_tests_graph_actions import unchanged_fields_feature_add


def check_feature_add(input_graph: torch_geometric.data.data.Data,
                      output_graph: torch_geometric.data.data.Data,
                      node_features: np.array,
                      last_feature_label: str):
    """
    Check the feature addition

    :param input_graph: Input graph before feature addition
    :param output_graph: Output graph after feature addition
    :param node_features: A numpy column containing the node feature for each node
    :param last_feature_label: The currently added feature label as string
    """

    # [1.] Field "x" of the output_graph will have an appended column --------------------------------------------------
    input_graph_x = input_graph.x.cpu().detach().numpy()
    output_graph_x = output_graph.x.cpu().detach().numpy()

    assert np.array_equal(node_features[:, 0], output_graph_x[:, -1]), "The last column of the \"x\" field of the " \
                                                                       "output graph must equal the \"node_features\"" \
                                                                       "of the input graph."

    assert np.array_equal(input_graph_x, output_graph_x[:, :-1]), "All columns of the \"x\" field of the output " \
                                                                  "graph except the last one, must equal the columns " \
                                                                  "of the input graph."

    # [2.] "edge_index", "edge_attr", "y", "node_labels", "node_ids", "edge_ids", "edge_attr_labels", "pos", "graph_id"
    #      stay intact - don't change ----------------------------------------------------------------------------------
    unchanged_fields_feature_add(input_graph, output_graph)

    # [3.] "node_feature_labels" change accordingly --------------------------------------------------------------------
    assert output_graph.node_feature_labels[-1] == last_feature_label, \
        f"The last feature label of the output graph's feature labels must be: {last_feature_label}."

    assert input_graph.node_feature_labels == output_graph.node_feature_labels[:-1], \
        "All elements of the \"node_feature_labels\" field of the output graph except the las one, must equal the " \
        "elements of the \"node_feature_labels\" field of the input graph"


########################################################################################################################
# MAIN Test ============================================================================================================
########################################################################################################################
def test_unit_add_feature():
    """
    Unit test add feature
    """

    ####################################################################################################################
    # [1.] Import graph data  ==========================================================================================
    ####################################################################################################################
    dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
    dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))

    dataset_len = len(dataset)
    graph_idx = random.randint(0, dataset_len - 1)
    input_graph = dataset[graph_idx]

    ####################################################################################################################
    # [2.] Add features to the graph ===================================================================================
    ####################################################################################################################
    feature_additions_nr = 10
    for feature_addition in range(feature_additions_nr):

        feature_label = "label_" + str(uuid.uuid4())
        node_size = input_graph.x.size(dim=0)

        node_features = np.random.randn(node_size, 1).astype(np.float32)    # one new feature for each node ------------

        output_graph = add_feature_all_nodes(input_graph, node_features, feature_label)

        # [2.1.] Check that the feature addition is successful ---------------------------------------------------------
        check_feature_add(input_graph, output_graph, node_features, feature_label)

        # [2.2.] Copy and repeat ---------------------------------------------------------------------------------------
        input_graph = output_graph
