"""
    Simulate the addition and removal of nodes

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-04-25
"""

import copy
import os
import pickle
import pytest
import random
import uuid

from hypothesis import given, settings
from hypothesis.strategies import integers
import numpy as np

from actionable.graph_actions import add_edge, remove_edge
from tests.utils_tests.utils_tests_graph_actions.utilities_for_tests_graph_actions import \
    unchanged_fields_edge_add_remove


@given(edge_additions_nr=integers(min_value=1, max_value=10))
@settings(max_examples=10, deadline=None)
def test_property_add_edges(edge_additions_nr: int):
    """
    Property-based test add edges

    :param edge_additions_nr: Number of edges that will be added
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

    # [1.2.] Add edges simulation --------------------------------------------------------------------------------------
    for edge in range(edge_additions_nr):

        # [1.3.] Get "edge_index" and then sample from the nodes for an edge that does not exist -----------------------
        nodes_nr = input_graph.x.size(dim=0)
        nodes_indexes_list = list(range(nodes_nr))

        input_graph_edge_index = input_graph.edge_index.cpu().detach().numpy()
        input_graph_edge_index_left = list(input_graph_edge_index[0, :])
        input_graph_edge_index_right = list(input_graph_edge_index[1, :])
        graph_edge_pairs = list(map(lambda x: (x[0], x[1]),
                                    list(zip(input_graph_edge_index_left, input_graph_edge_index_right))))

        edge_index_left = random.choice(nodes_indexes_list)
        edge_index_right = random.choice(nodes_indexes_list)
        while edge_index_left == edge_index_right:
            edge_index_right = random.choice(nodes_indexes_list)

        new_edge_attr = None

        # [1.4.] The edge does not exist already -----------------------------------------------------------------------
        if not (edge_index_left, edge_index_right) in graph_edge_pairs and \
                not (edge_index_right, edge_index_left) in graph_edge_pairs:

            output_graph = add_edge(input_graph, edge_index_left, edge_index_right, new_edge_attr)

            # [1.4.1.] Fields that don't change ------------------------------------------------------------------------
            unchanged_fields_edge_add_remove(input_graph, output_graph)

            # [1.4.2.] Check the properties at the end of the edge additions -------------------------------------------
            assert input_graph.edge_index.size(dim=1) + 1 == output_graph.edge_index.size(dim=1), \
                f"The columns of the features \"edge_index\" must be increased exactly by one edge that is added."
            assert input_graph.edge_index.size(dim=0) == output_graph.edge_index.size(dim=0), \
                f"The rows of the features \"edge_index\", should not change"

            assert len(input_graph.edge_ids) + 1 == len(output_graph.edge_ids), \
                f"The length of the \"edge_ids\" must be increased exactly by one."

            if input_graph.edge_attr is None:
                assert output_graph.edge_attr is None, "If the input graph's \"edge_attr\" is None, then after an " \
                                                       "edge addition should keep the \"edge_attr\" as None."
            else:
                assert input_graph.edge_attr.size(dim=0) + 1 == output_graph.edge_attr.size(dim=0), \
                    f"The rows of the features \"edge_attr\" must be increased exactly by one edge that is added."
                assert input_graph.edge_attr.size(dim=1) == output_graph.edge_attr.size(dim=1), \
                    f"The columns of the features \"edge_attr\", should not change"

            # [1.4.3.] Copy and repeat ---------------------------------------------------------------------------------
            input_graph = output_graph

        # [1.5.] The edge already exist - Expected exception must be thrown --------------------------------------------
        else:
            with pytest.raises(ValueError) as excinfo:
                add_edge(input_graph, edge_index_left, edge_index_right, new_edge_attr)


"""
@given(edge_additions_nr=integers(min_value=1, max_value=10))
@settings(max_examples=10, deadline=None)
def test_property_remove_edges(edge_additions_nr: int):
    # Property-based test remove edges

    # :param edge_additions_nr: Number of edges that will be removed


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
"""
