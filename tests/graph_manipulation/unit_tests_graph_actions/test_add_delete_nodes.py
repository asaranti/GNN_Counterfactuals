"""
    Test add node(s)

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-04-11
"""

import os
import pickle
import random
import torch
import uuid

import numpy as np
import torch_geometric
from torch_geometric.data import Data

from actionable.graph_actions import add_node, remove_node


def check_node_add(input_graph: torch_geometric.data.data.Data,
                   output_graph: torch_geometric.data.data.Data,
                   node_features: np.array,
                   node_label: str,
                   node_id: str):
    """
    Check the node additions

    :param input_graph: Input graph before node addition
    :param output_graph: Output graph after node addition
    :param node_features: A numpy row containing the node features
    :param node_label: Node label, which will be added
    :param node_id: Node id, which will be added
    """

    # [1.] Field "x" of the output_graph will have an appended column --------------------------------------------------
    input_graph_x = input_graph.x.cpu().detach().numpy()
    output_graph_x = output_graph.x.cpu().detach().numpy()

    assert np.array_equal(node_features[0], output_graph_x[-1, :]), "The last column of the \"x\" field of the " \
                                                                    "output graph must equal the \"node_features\"" \
                                                                    "of the input graph."

    assert np.array_equal(input_graph_x, output_graph_x[:-1, :]), "All columns of the \"x\" field of the output " \
                                                                  "graph except the last one, must equal the columns " \
                                                                  "of the input graph."

    # [2.] "edge_index", "edge_attr", "y", "edge_ids", "edge_attr_labels", "pos", "graph_id", "node_feature_labels" ----
    #      stay intact - don't change ----------------------------------------------------------------------------------
    assert torch.equal(input_graph.edge_index, output_graph.edge_index), \
        "The input's and output's graph \"edge_index\" fields must be equal."
    if input_graph.edge_attr is None:
        assert output_graph.edge_attr is None, "If the input graph's \"edge_attr\" is None, then after a node " \
                                                "addition should keep the \"edge_attr\" as None."
    else:
        assert torch.equal(input_graph.edge_attr, output_graph.edge_attr), \
            "The input's and output's graph \"edge_attr\" fields must be equal."
    assert torch.equal(input_graph.y, output_graph.y), "The input's and output's graph \"y\" fields must be equal."
    assert input_graph.edge_ids == output_graph.edge_ids, "The input's and output's graph \"edge_ids\" fields must " \
                                                          "be equal."
    assert input_graph.edge_attr_labels == output_graph.edge_attr_labels, \
        "The input's and output's graph \"edge_attr_labels\" fields must be equal."
    assert input_graph.pos == output_graph.pos, "The input's and output's graph \"pos\" fields must be equal."
    assert input_graph.graph_id == output_graph.graph_id, "The input's and output's graph \"graph_id\" fields must " \
                                                          "be equal."

    # [3.] "node_labels", "node_ids" change accordingly ----------------------------------------------------------------
    assert output_graph.node_labels[-1] == node_label, f"The last node label of the output graph's node labels must " \
                                                       f"be: {node_label}."
    assert np.array_equal(input_graph.node_labels, output_graph.node_labels[:-1]), \
        "All elements of the \"node_labels\" field of the output graph except the last one, must equal the elements " \
        "of the \"node_labels\" field of the input graph."

    assert output_graph.node_ids[-1] == node_id, f"The last node id of the output graph's node labels must " \
                                                 f"be: {node_id}."
    assert np.array_equal(input_graph.node_ids, output_graph.node_ids[:-1]), \
        "All elements of the \"node_ids\" field of the output graph except the last one, must equal the elements " \
        "of the \"node_ids\" field of the input graph."

    # TODO >>> dtype >>>>>>>>>>>>>>>>>>>>>>>>>>>


########################################################################################################################
# MAIN Test ============================================================================================================
########################################################################################################################
def test_unit_add_nodes():
    """
    Unit test add nodes
    """

    # [1.] Transformation Experiment ::: From PPI to Pytorch_Graph -----------------------------------------------------
    dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
    dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))

    dataset_len = len(dataset)
    graph_idx = random.randint(0, dataset_len)
    input_graph = dataset[graph_idx]

    # [2.] Try node addition(s) ----------------------------------------------------------------------------------------
    node_id = str(uuid.uuid4())
    label = "label_" + str(uuid.uuid4())
    node_features_size = input_graph.x.size(dim=1)

    node_features = np.random.randn(1, node_features_size).astype(np.float32)

    output_graph = add_node(input_graph, node_features, label, node_id)

    # [3.] Check that the node addition is successful ------------------------------------------------------------------
    check_node_add(input_graph, output_graph, node_features, label, node_id)
