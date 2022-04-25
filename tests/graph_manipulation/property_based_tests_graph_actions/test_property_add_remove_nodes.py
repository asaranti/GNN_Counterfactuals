"""
    Simulate the addition and removal of nodes

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-11-10
"""

import copy
import os
import pickle
import random
import uuid

from hypothesis import given, settings
from hypothesis.strategies import integers
import numpy as np

from actionable.graph_actions import add_node, remove_node
from tests.utils_tests.utils_tests_graph_actions.utilities_for_tests_graph_actions import unchanged_fields_node_add, \
    unchanged_fields_node_remove


@given(node_additions_nr=integers(min_value=1, max_value=10))
@settings(max_examples=10, deadline=None)
def test_property_add_nodes(node_additions_nr: int):
    """
    Property-based test add nodes

    :param node_additions_nr: Number of nodes that will be added
    """

    ####################################################################################################################
    # [1.] Import graph data and add the nodes =========================================================================
    ####################################################################################################################
    # [1.1.] Import graph data -----------------------------------------------------------------------------------------
    dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
    dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))

    dataset_len = len(dataset)
    graph_idx = random.randint(0, dataset_len - 1)
    input_graph = dataset[graph_idx]
    input_graph_original = copy.deepcopy(input_graph)

    # [1.2.] Add nodes simulation --------------------------------------------------------------------------------------
    for node_addition in range(node_additions_nr):

        node_id = str(uuid.uuid4())
        label = "label_" + str(uuid.uuid4())
        node_features_size = input_graph.x.size(dim=1)

        node_features = np.random.randn(1, node_features_size).astype(np.float32)

        output_graph = add_node(input_graph, node_features, label, node_id)

        # [1.3.] Copy and repeat ---------------------------------------------------------------------------------------
        input_graph = output_graph

    ####################################################################################################################
    # [2.] Check the properties at the end of the node additions =======================================================
    ####################################################################################################################
    # [2.1.] Field "x" of the last graph will have node_additions_nr appended rows -------------------------------------
    assert input_graph_original.x.size(dim=0) + node_additions_nr == input_graph.x.size(dim=0), \
        f"The rows of the features \"x\" must be increased exactly by the number of added nodes."
    assert input_graph_original.x.size(dim=1) == input_graph.x.size(dim=1), \
        f"The columns of the features \"x\", should not change"

    # [2.2.] Fields that remain unchanged ------------------------------------------------------------------------------
    unchanged_fields_node_add(input_graph_original, input_graph)

    # [2.3.] "node_labels", "node_ids" change accordingly --------------------------------------------------------------
    assert len(input_graph_original.node_labels) + node_additions_nr == len(input_graph.node_labels), \
        f"The length of the \"node_labels\" must be increased exactly by the number of added nodes."
    assert len(input_graph_original.node_ids) + node_additions_nr == len(input_graph.node_ids), \
        f"The length of the \"node_ids\" must be increased exactly by the number of added nodes."


@given(node_removals_nr=integers(min_value=1, max_value=10))
@settings(max_examples=10, deadline=None)
def test_property_remove_nodes(node_removals_nr: int):
    """
    Property-based test remove nodes

    :param node_removals_nr: Number of nodes that will be removed
    """

    ####################################################################################################################
    # [1.] Import graph data and remove the nodes ======================================================================
    ####################################################################################################################
    # [1.1.] Transformation Experiment ::: From PPI to Pytorch_Graph ---------------------------------------------------
    dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
    dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))

    dataset_len = len(dataset)
    graph_idx = random.randint(0, dataset_len - 1)
    input_graph = dataset[graph_idx]
    input_graph_original = copy.deepcopy(input_graph)

    # [1.2.] Try node addition(s) --------------------------------------------------------------------------------------
    for node_removal in range(node_removals_nr):

        nodes_nr = input_graph.x.size(dim=0)
        node_index = random.randint(0, nodes_nr - 1)
        output_graph = remove_node(input_graph, node_index)

        # [1.3.] Copy and repeat ---------------------------------------------------------------------------------------
        input_graph = output_graph

    ####################################################################################################################
    # [2.] Check the properties at the end of the node removals ========================================================
    ####################################################################################################################
    # [2.1.] Field "x" of the last graph will have node_removals_nr deleted rows ---------------------------------------
    print(input_graph_original.x.size(dim=0), node_removals_nr, input_graph.x.size(dim=0))
    assert input_graph_original.x.size(dim=0) - node_removals_nr == input_graph.x.size(dim=0), \
        f"The rows of the features \"x\" must be decreased exactly by the number of removed nodes."
    assert input_graph_original.x.size(dim=1) == input_graph.x.size(dim=1), \
        f"The columns of the features \"x\", should not change."

    # [2.2.] Unchanged fields of node removal --------------------------------------------------------------------------
    unchanged_fields_node_remove(input_graph_original, input_graph)

    # [2.3.] "edge_index", "edge_attr", "edge_ids" will probably change ------------------------------------------------
    assert input_graph_original.edge_index.size(dim=0) == input_graph.edge_index.size(dim=0), \
        f"The number of \"edge_index\" rows must be equal with the original graph."
    assert input_graph_original.edge_index.size(dim=1) >= input_graph.edge_index.size(dim=1), \
        f"The number of edges must necessarily be equal or less with the original graph."

    if input_graph_original.edge_attr is not None:
        assert input_graph_original.edge_attr.size(dim=0) >= input_graph.edge_attr.size(dim=0), \
            f"The number of \"edge_attr\" rows must necessarily be equal or less with the original graph."
        assert input_graph_original.edge_attr.size(dim=1) == input_graph.edge_attr.size(dim=1), \
            f"The number of \"edge_attr\" columns must be equal with the original graph."

    assert len(input_graph_original.edge_ids) >= len(input_graph.edge_ids), \
        f"The length of the \"edge_ids\" must necessarily be equal or less with the original graph"

    # [2.4.] "node_labels", "node_ids" change accordingly --------------------------------------------------------------
    assert len(input_graph_original.node_labels) - node_removals_nr == len(input_graph.node_labels), \
        f"The length of the \"node_labels\" must be increased exactly by the number of added nodes."
    assert len(input_graph_original.node_ids) - node_removals_nr == len(input_graph.node_ids), \
        f"The length of the \"node_ids\" must be increased exactly by the number of added nodes."

