"""
    Test add edge(s)

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-04-19
"""

import os
import pickle
import random
import pytest
from typing import Optional
import uuid

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

from actionable.graph_actions import add_edge


def check_edge_add(input_graph: torch_geometric.data.data.Data,
                   output_graph: torch_geometric.data.data.Data,
                   new_edge_index_left: int,
                   new_edge_index_right: int,
                   new_edge_attr: Optional[np.ndarray]
                   ):
    """
    Check the edge addition

    :param input_graph: Input graph before node addition
    :param output_graph: Output graph after node addition
    :param new_edge_index_left: Index of left node of new edge
    :param new_edge_index_right: Index of right node of new edge
    :param new_edge_attr: Attribute(s) of new edge
    """

    # [1.] Fields that don't change ------------------------------------------------------------------------------------
    assert torch.equal(input_graph.x, output_graph.x), "The input's and output's graph \"x\" fields must be equal."
    assert torch.equal(input_graph.y, output_graph.y), "The input's and output's graph \"y\" fields must be equal."
    assert input_graph.node_labels == output_graph.node_labels, \
        "The input's and output's graph \"node_labels\" fields must be equal."
    assert input_graph.node_ids == output_graph.node_ids, \
        "The input's and output's graph \"node_ids\" fields must be equal."
    assert input_graph.node_feature_labels == output_graph.node_feature_labels, \
        "The input's and output's graph \"node_feature_labels\" fields must be equal."
    assert input_graph.edge_attr_labels == output_graph.edge_attr_labels, \
        "The input's and output's graph \"edge_attr_labels\" fields must be equal."
    assert input_graph.pos == output_graph.pos, "The input's and output's graph \"pos\" fields must be equal."
    assert input_graph.graph_id == output_graph.graph_id, "The input's and output's graph \"graph_id\" " \
                                                          "fields must be equal."

    # [2.] "egde_index", "edge_ids", "edge_attr" -----------------------------------------------------------------------
    input_graph_edge_index = input_graph.edge_index.cpu().detach().numpy()
    output_graph_edge_index = output_graph.edge_index.cpu().detach().numpy()
    edge_id_array = np.array([new_edge_index_left, new_edge_index_right])
    edge_id = f"{new_edge_index_left}_{new_edge_index_right}"

    # [2.1.] "egde_index" ----------------------------------------------------------------------------------------------
    assert np.array_equal(edge_id_array, output_graph_edge_index[:, -1]), \
        f"The last column of the \"egde_index\" field of the output graph must equal the edge of the given nodes: " \
        f"{new_edge_index_left} - {new_edge_index_right}."

    assert np.array_equal(input_graph_edge_index, output_graph_edge_index[:, :-1]), \
        "All columns of the \"edge_index\" field of the output graph except the last one, must equal the columns " \
        "of the input graph."

    # [2.2.] "edge_ids" ------------------------------------------------------------------------------------------------
    assert output_graph.edge_ids[-1] == edge_id, f"The last node id of the output graph's node labels must " \
                                                 f"be: {edge_id}."
    assert input_graph.edge_ids == output_graph.edge_ids[:-1], \
        "All elements of the \"edge_ids\" field of the output graph except the last one, must equal the elements " \
        "of the \"edge_ids\" field of the input graph."

    # [2.3.] "edge_attr" -----------------------------------------------------------------------------------------------
    if input_graph.edge_attr is None:
        assert output_graph.edge_attr is None, "If the input graph's \"edge_attr\" is None, then after a node " \
                                          "addition should keep the \"edge_attr\" as None."
    else:
        input_graph_edge_attr = input_graph.edge_attr.cpu().detach().numpy()
        output_graph_edge_attr = output_graph.edge_attr.cpu().detach().numpy()

        assert np.array_equal(new_edge_attr, output_graph_edge_attr[-1, :]), \
            "The last row of the \"edge_attr\" field of the output graph must equal the \"new_edge_attr\"" \
            "of the input graph."

        assert np.array_equal(input_graph_edge_attr, output_graph_edge_attr[:-1, :]), \
            "All rows of the \"edge_attr\" field of the output graph except the last one, must equal the columns " \
            "of the input graph."


# MAIN Tests ===========================================================================================================
def test_unit_add_edge_correct_kirc_random():
    """
    Unit test add nodes that is expected to be correct
    """

    # [1.] Transformation Experiment ::: From PPI to Pytorch_Graph -----------------------------------------------------
    dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
    dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))

    dataset_len = len(dataset)
    graph_idx = random.randint(0, dataset_len - 1)
    input_graph = dataset[graph_idx]

    # [2.] Valid edge addition =========================================================================================
    edge_node_left = 0
    edge_node_right = 1
    new_edge_attr = None
    output_graph = add_edge(input_graph, edge_node_left, edge_node_right, new_edge_attr)

    # Check that the node addition is successful -----------------------------------------------------------------------
    check_edge_add(input_graph, output_graph, edge_node_left, edge_node_right, new_edge_attr)


def test_unit_add_edge_expected_exception_kirc_random():
    """
    Unit test add nodes that is expected to be wrong and raise an exception
    """

    # [1.] Transformation Experiment ::: From PPI to Pytorch_Graph -----------------------------------------------------
    dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
    dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))

    dataset_len = len(dataset)
    graph_idx = random.randint(0, dataset_len - 1)
    input_graph = dataset[graph_idx]

    # [2.] Non-valid edge addition =====================================================================================
    input_graph_edge_index = input_graph.edge_index.cpu().detach().numpy()
    edges_nr = input_graph_edge_index.shape[1]
    selected_edge_idx = random.randint(0, edges_nr - 1)
    selected_edge = input_graph_edge_index[:, selected_edge_idx]
    new_edge_attr = None

    with pytest.raises(ValueError) as excinfo:
        output_graph = add_edge(input_graph, selected_edge[0], selected_edge[1], new_edge_attr)


def test_unit_add_edge_small_graph():
    """
    Unit test that is expected to be correct
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

    input_graph_1_2 = Data(x=torch.from_numpy(node_features_1_2),
                           edge_index=torch.Tensor(2, 0),
                           edge_attr=None,
                           y=torch.from_numpy(np.array([1])),
                           node_labels=[node_label_1, node_label_2],
                           node_ids=[node_id_1, node_id_2],
                           node_feature_labels=node_feature_labels,
                           edge_ids=[],
                           edge_attr_labels=edge_feature_labels,
                           pos=None,
                           graph_id=graph_id
                           )

    # [2.] Valid edge addition -----------------------------------------------------------------------------------------
    edge_node_left = 0
    edge_node_right = 1
    new_edge_attr = np.random.randn(1, edge_features_size).astype(np.float32)

    graph_3 = add_edge(input_graph_1_2, edge_node_left, edge_node_right, new_edge_attr)

    # [3.] Try again to re-enter the edde, this should raise an exception ----------------------------------------------
    with pytest.raises(ValueError) as excinfo:
        graph_4 = add_edge(graph_3, edge_node_left, edge_node_right, new_edge_attr)

