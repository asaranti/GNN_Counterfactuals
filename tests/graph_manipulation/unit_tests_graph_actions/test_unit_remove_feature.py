"""
    Test remove feature(s)

    :author: David Kerschbaumer
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-05-27
"""

import os
import pickle
import random
import uuid

import numpy as np
import torch_geometric

from actionable.graph_actions import add_feature_all_nodes, remove_feature_all_nodes
from tests.utils_tests.utils_tests_graph_actions.utilities_for_tests_graph_actions import \
    unchanged_fields_feature_remove


def check_feature_remove(input_graph: torch_geometric.data.data.Data,
                         output_graph: torch_geometric.data.data.Data,
                         feature_index_to_remove: int):
    """
    Check the feature removal

    :param input_graph: Input graph before feature addition
    :param output_graph: Output graph after feature addition
    :param feature_index_to_remove: A integer containing the index of the feature to remove
    """

    # [1.] Field "x" of the output_graph will have an appended column --------------------------------------------------
    input_graph_x = input_graph.x.cpu().detach().numpy()
    output_graph_x = output_graph.x.cpu().detach().numpy()

    assert np.array_equal(np.delete(input_graph_x, feature_index_to_remove, 1), output_graph_x), \
        f"The output graph's \"x\" field must have the corresponding column with index: " \
        f"{feature_index_to_remove} removed."

    assert input_graph.node_feature_labels[:feature_index_to_remove] + \
           input_graph.node_feature_labels[feature_index_to_remove:] == output_graph.node_feature_labels, \
           "The output's graph \"node_feature_labels\" must have the \"node_feature_label\" " \
           "at the \"feature_index_to_remove\" removed."

    # [2.] "y", "pos", "graph_id", "node_feature_labels", "edge_attr_labels" stay intact - don't change ----------------
    unchanged_fields_feature_remove(input_graph, output_graph)


########################################################################################################################
# MAIN Test ============================================================================================================
########################################################################################################################
def test_unit_remove_feature():
    """
    Unit test remove feature
    """

    ####################################################################################################################
    # [1.] Import graph data ===========================================================================================
    ####################################################################################################################
    dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
    dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))

    dataset_len = len(dataset)
    graph_idx = random.randint(0, dataset_len - 1)
    input_graph = dataset[graph_idx]

    ####################################################################################################################
    # [2.] Add features to the graph which can be removed in [3.] ======================================================
    ####################################################################################################################
    # [2.1.] Create test-list containing all feature labels  -----------------------------------------------------------
    added_feature_label_list = [x for x in input_graph.node_feature_labels]

    # [2.2.] add feature_removal_nr times features ---------------------------------------------------------------------
    feature_additions_nr = 10
    for feature_addition in range(feature_additions_nr):
        feature_label = "label_" + str(uuid.uuid4())
        added_feature_label_list.append(feature_label)  # add new created features to the list -------------------------
        node_size = input_graph.x.size(dim=0)
        node_features = np.random.randn(node_size, 1).astype(np.float32)    # one new feature for each node ------------
        output_graph = add_feature_all_nodes(input_graph, node_features, feature_label)
        input_graph = output_graph

    ####################################################################################################################
    # [3.] Remove feature_additions_nr times a features from input_graph ===============================================
    ####################################################################################################################
    for feature_removal in range(feature_additions_nr):

        feature_nr = input_graph.x.size(dim=1)
        feature_index_to_remove = random.randint(0, feature_nr - 1)
        del added_feature_label_list[feature_index_to_remove] # remove the feature also from our test list

        output_graph = remove_feature_all_nodes(input_graph, feature_index_to_remove)

        # [3.1] Check that the feature addition is successful ----------------------------------------------------------
        check_feature_remove(input_graph, output_graph, feature_index_to_remove)

        # [3.2] Copy and repeat ----------------------------------------------------------------------------------------
        input_graph = output_graph

    ####################################################################################################################
    # [4.] Check the properties at the end of the feature removals =====================================================
    ####################################################################################################################
    # [4.1.] Check if all removed node_feature_labels are really removed -----------------------------------------------
    assert input_graph.node_feature_labels == added_feature_label_list, \
        "The final graph's \"node_feature_labels\" must only contain \"node_feature_label\"'s which were not removed"
