"""
    Test remove feature(s)

    :author: David Kerschbaumer
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-06-13
"""

import copy
import os
import pickle
import random
import uuid

from hypothesis import given, settings
from hypothesis.strategies import integers
import numpy as np

from actionable.graph_actions import add_feature, remove_feature
from tests.utils_tests.utils_tests_graph_actions.utilities_for_tests_graph_actions import unchanged_fields_feature_remove, \
    unchanged_fields_feature_add


@given(feature_additions_nr=integers(min_value=1, max_value=10))
@settings(max_examples=10, deadline=None)
def test_property_add_features(feature_additions_nr: int):
    """
    Property-based test add features

    :param feature_additions_nr: Number of features that will be added
    """

    ####################################################################################################################
    # [1.] Import graph data  ==========================================================================================
    ####################################################################################################################
    dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
    dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))

    dataset_len = len(dataset)
    graph_idx = random.randint(0, dataset_len - 1)
    input_graph = dataset[graph_idx]
    input_graph_original = copy.deepcopy(input_graph)

    ####################################################################################################################
    # [2.] Add features to the graph ===================================================================================
    ####################################################################################################################
    for feature_addition in range(feature_additions_nr):

        feature_label = "label_" + str(uuid.uuid4())
        node_size = input_graph.x.size(dim=0)

        node_features = np.random.randn(node_size, 1).astype(np.float32)  # one new feature for each node

        output_graph = add_feature(input_graph, node_features, feature_label)

        # [2.1.] Copy and repeat ---------------------------------------------------------------------------------------
        input_graph = output_graph

    ####################################################################################################################
    # [3.] Check the properties at the end of the feature additions ====================================================
    ####################################################################################################################
    # [3.1.] Field "x" of the last graph will have feature_additions_nr appended columns -------------------------------
    assert input_graph_original.x.size(dim=1) + feature_additions_nr == input_graph.x.size(dim=1), \
        f"The columns (the features) of \"x\" must be increased exactly by the number of added features."
    assert input_graph_original.x.size(dim=0) == input_graph.x.size(dim=0), \
        f"The rows (the nodes) of \"x\", should not change"

    # [3.2.] Fields that remain unchanged ------------------------------------------------------------------------------
    unchanged_fields_feature_add(input_graph_original, input_graph)

    # [3.3.] "node_feature_labels" change accordingly ------------------------------------------------------------------
    assert len(input_graph_original.node_feature_labels) + feature_additions_nr == \
           len(input_graph.node_feature_labels), \
        f"The length of the \"node_feature_labels\" must be increased exactly by the number of added features."






@given(feature_removals_nr=integers(min_value=1, max_value=10))
@settings(max_examples=10, deadline=None)
def test_property_remove_features(feature_removals_nr: int):
    """
    Property-based test remove features

    :param feature_removals_nr: Number of features that will be first added and then removed
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
    added_feature_label_list = [x for x in input_graph.node_feature_labels] # track all the labels from features

    # [2.2.] add feature_removal_nr times features ---------------------------------------------------------------------
    for feature_addition in range(feature_removals_nr):
        feature_label = "label_" + str(uuid.uuid4())
        added_feature_label_list.append(feature_label) # add the new created features to the test-list
        node_size = input_graph.x.size(dim=0)
        node_features = np.random.randn(node_size, 1).astype(np.float32) # one new feature for each node

        output_graph = add_feature(input_graph, node_features, feature_label)
        input_graph = output_graph

    # [2.3.] copy the new created graph for checks in [4.]  ------------------------------------------------------------
    input_graph_original = copy.deepcopy(input_graph)

    ####################################################################################################################
    # [3.] Remove feature_removals_nr times a features from input_graph ================================================
    ####################################################################################################################
    for feature_removal in range(feature_removals_nr):

        feature_nr = input_graph.x.size(dim=1)
        feature_index_to_remove = random.randint(0, feature_nr - 1)
        del added_feature_label_list[feature_index_to_remove] # remove the feature also from our test-list

        output_graph = remove_feature(input_graph, feature_index_to_remove)

        # [3.3.] Copy and repeat ---------------------------------------------------------------------------------------
        input_graph = output_graph


    ####################################################################################################################
    # [4.] Check the properties at the end of the feature removals =====================================================
    ####################################################################################################################
    # [4.1.] Field "x" of the last graph will have feature_removals_nr deleted columns ---------------------------------

    assert input_graph_original.x.size(dim=1) - feature_removals_nr == input_graph.x.size(dim=1), \
        f"The columns (the features) of \"x\" must be decreased exactly by the number of removed features."
    assert input_graph_original.x.size(dim=0) == input_graph.x.size(dim=0), \
        f"The rows (the nodes) of \"x\", should not change."

    # [4.2.] Unchanged fields of feature removal -----------------------------------------------------------------------
    unchanged_fields_feature_remove(input_graph_original, input_graph)

    # [4.3.] "node_feature_labels" change accordingly --------------------------------------------------------------
    assert len(input_graph_original.node_feature_labels) - feature_removals_nr == len(input_graph.node_feature_labels),\
        f"The length of the \"node_feature_labels\" must be increased exactly by the number of added features."

    # [4.4] Check if all removed node_feature_labels are really removed ------------------------------------------------
    assert input_graph.node_feature_labels == added_feature_label_list, \
    "The final graph's \"node_feature_labels\" may only contain \"node_feature_label\"'s which were not removed"

