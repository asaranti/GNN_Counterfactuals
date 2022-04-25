"""
    Test remove edge(s)

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-04-25
"""

import copy
import os
import pickle
import random
import pytest
import uuid

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

from actionable.graph_actions import remove_edge
from tests.utils_tests.utils_tests_graph_actions.utilities_for_tests_graph_actions import \
    unchanged_fields_edge_add_remove


def check_edge_remove(input_graph: torch_geometric.data.data.Data,
                      output_graph: torch_geometric.data.data.Data,
                      selected_edge_idx: int):
    """
    Check the edge addition

    :param input_graph: Input graph before node addition
    :param output_graph: Output graph after node addition
    :param selected_edge_idx: Index of the edge that will be removed
    """

    # [1.] Fields that don't change ------------------------------------------------------------------------------------
    unchanged_fields_edge_add_remove(input_graph, output_graph)

    # [2.] "egde_index", "edge_ids", "edge_attr" -----------------------------------------------------------------------
    # [2.1.] "egde_index" ----------------------------------------------------------------------------------------------
    input_graph_edge_index = input_graph.edge_index.cpu().detach().numpy()
    output_graph_edge_index = output_graph.edge_index.cpu().detach().numpy()
    input_graph_edge_index_after_delete = np.delete(input_graph_edge_index, selected_edge_idx, axis=1)
    assert np.array_equal(input_graph_edge_index_after_delete,
                          output_graph_edge_index), "The \"egde_index\" must have exactly this removed index."

    # [2.2.] "edge_ids" ------------------------------------------------------------------------------------------------
    input_graph_edge_ids = input_graph.edge_ids
    output_graph_edge_ids = output_graph.edge_ids

    input_graph_edge_after_delete = copy.deepcopy(input_graph_edge_ids)
    del input_graph_edge_after_delete[selected_edge_idx]
    assert input_graph_edge_after_delete == output_graph_edge_ids, "The output graph's \"egde_ids\" must have " \
                                                                   "exactly the removed edge_id removed"

    # [2.3.] "edge_attr" -----------------------------------------------------------------------------------------------
    if input_graph.edge_attr is None:
        assert output_graph.edge_attr is None, "If the input graph's \"edge_attr\" is None, then after a node " \
                                          "addition should keep the \"edge_attr\" as None."
    else:
        input_graph_edge_attr = input_graph.edge_attr.cpu().detach().numpy()
        output_graph_edge_attr = output_graph.edge_attr.cpu().detach().numpy()

        input_graph_edge_attr_after_delete = np.delete(input_graph_edge_attr, selected_edge_idx, axis=1)
        assert np.array_equal(input_graph_edge_attr_after_delete,
                              output_graph_edge_attr), "The \"egde_attr\" must have exactly this removed index."


def test_unit_remove_edge_correct_expected_exc_kirc_random():
    """
    Unit test remove edge correctly and then try to remove it again, which supposed to raise an expected exception
    """

    # [1.] Transformation Experiment ::: From PPI to Pytorch_Graph -----------------------------------------------------
    dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
    dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))

    dataset_len = len(dataset)
    graph_idx = random.randint(0, dataset_len - 1)
    input_graph = dataset[graph_idx]

    # [2.] Edge removal ------------------------------------------------------------------------------------------------
    input_graph_edge_index = input_graph.edge_index.cpu().detach().numpy()
    edges_nr = input_graph_edge_index.shape[1]
    selected_edge_idx = random.randint(0, edges_nr - 1)
    selected_edge = input_graph_edge_index[:, selected_edge_idx]
    selected_edge_left = selected_edge[0]
    selected_edge_right = selected_edge[1]

    # [3.] Remove the selected edge ------------------------------------------------------------------------------------
    output_graph = remove_edge(input_graph, selected_edge_left, selected_edge_right)

    # [4.] Check the edge removal correctness --------------------------------------------------------------------------
    check_edge_remove(input_graph, output_graph, selected_edge_idx)

    # [5.] Try to remove the edge again, this should raise an Exception ------------------------------------------------
    with pytest.raises(ValueError) as excinfo:
        remove_edge(output_graph, selected_edge_left, selected_edge_right)


def test_unit_remove_many_edges_correct_expected_exc_kirc_random():
    """
    Unit test remove many edges correctly and then try to remove it again,
    which supposed to raise an expected exception
    """

    # [1.] Transformation Experiment ::: From PPI to Pytorch_Graph -----------------------------------------------------
    dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
    dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))

    dataset_len = len(dataset)
    graph_idx = random.randint(0, dataset_len - 1)
    input_graph = dataset[graph_idx]

    edge_removals_nr = 10

    for edge_removal_try in range(edge_removals_nr):

        # [2.] Edge removal --------------------------------------------------------------------------------------------
        input_graph_edge_index = input_graph.edge_index.cpu().detach().numpy()
        edges_nr = input_graph_edge_index.shape[1]
        selected_edge_idx = random.randint(0, edges_nr - 1)
        selected_edge = input_graph_edge_index[:, selected_edge_idx]
        selected_edge_left = selected_edge[0]
        selected_edge_right = selected_edge[1]

        # [3.] Remove the selected edge --------------------------------------------------------------------------------
        output_graph = remove_edge(input_graph, selected_edge_left, selected_edge_right)

        # [4.] Check the edge removal correctness ----------------------------------------------------------------------
        check_edge_remove(input_graph, output_graph, selected_edge_idx)

        # [5.] Try to remove the edge again, this should raise an Exception --------------------------------------------
        with pytest.raises(ValueError) as excinfo:
            remove_edge(output_graph, selected_edge_left, selected_edge_right)

        # [6.] Replace input graph with output graph -------------------------------------------------------------------
        input_graph = output_graph


def test_unit_remove_edge_small_graph():
    """
    Unit test remove edge that is expected to be correct
    """

    # [1.] Input graph with two nodes ----------------------------------------------------------------------------------
    node_features_size = 2
    edge_features_size = 3

    node_feature_labels = ["node_feature_1", "node_feature_2"]
    edge_feature_labels = ["edge_feature_1", "edge_feature_2", "edge_feature_3"]

    node_id_1 = str(uuid.uuid4())
    node_label_1 = "label_" + str(uuid.uuid4())

    node_id_2 = str(uuid.uuid4())
    node_label_2 = "label_" + str(uuid.uuid4())
    node_features_1_2 = np.random.randn(2, node_features_size).astype(np.float32)

    graph_id = "graph_" + str(uuid.uuid4())

    # [2.] Insert the edge ---------------------------------------------------------------------------------------------
    edge_index = np.array([[0], [1]])
    edge_ids = ["edge_0_1"]
    edge_attr = np.random.randn(1, edge_features_size).astype(np.float32)

    input_graph_1_2 = Data(x=torch.from_numpy(node_features_1_2),
                           edge_index=torch.from_numpy(edge_index),
                           edge_attr=torch.from_numpy(edge_attr),
                           y=torch.from_numpy(np.array([1])),
                           node_labels=[node_label_1, node_label_2],
                           node_ids=[node_id_1, node_id_2],
                           node_feature_labels=node_feature_labels,
                           edge_ids=edge_ids,
                           edge_attr_labels=edge_feature_labels,
                           pos=None,
                           graph_id=graph_id
                           )

    # [3.] Remove the selected edge ------------------------------------------------------------------------------------
    selected_edge_left = 0
    selected_edge_right = 1
    output_graph = remove_edge(input_graph_1_2, selected_edge_left, selected_edge_right)

    # [4.] Fields that don't change ------------------------------------------------------------------------------------
    unchanged_fields_edge_add_remove(input_graph_1_2, output_graph)

    # [5.] "egde_index", "edge_ids", "edge_attr" -----------------------------------------------------------------------
    assert torch.equal(output_graph.edge_index, torch.Tensor(2, 0).to(dtype=torch.int64)), \
        "The \"edge_index\" is expected to be empty."
    assert output_graph.edge_ids == [], "The \"edge_ids\" is expected to be empty."
    assert torch.equal(output_graph.edge_attr, torch.Tensor(0, edge_features_size)), "The \"edge_attr\" is expected " \
                                                                                     "to be empty."

    # [6.] Try to again remove the already removed edge and catch an expected exception --------------------------------
    with pytest.raises(ValueError) as excinfo:
        remove_edge(output_graph, selected_edge_left, selected_edge_right)

