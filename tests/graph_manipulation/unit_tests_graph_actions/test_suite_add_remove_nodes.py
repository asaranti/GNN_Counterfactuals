"""
    Test suite add + remove node(s)

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-04-12
"""

import copy
import random
import uuid

import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

from actionable.graph_actions import add_node, remove_node


def unchanged_fields_node_add_remove(graph_1: Data, graph_2: Data):
    """
    Make the checks for the node and add remove, for the fields
    that are not supposed to change after the node add and remove

    :param graph_1: Input graph
    :param graph_2: Output graph
    """

    assert torch.equal(graph_1.edge_index, graph_2.edge_index), \
        "The input's and output's graph \"edge_index\" fields must be equal."
    if graph_1.edge_attr is None:
        assert graph_2.edge_attr is None, "The input's and output's graph \"edge_attr\" fields must be equally None."
    else:
        assert torch.equal(graph_1.edge_attr, graph_2.edge_attr), \
            "The input's and output's graph \"edge_attr\" fields must be equal."
    assert torch.equal(graph_1.y, graph_2.y), "The input's and output's graph \"y\" fields must be equal."
    assert graph_1.node_feature_labels == graph_2.node_feature_labels, \
        "The input's and output's graph \"node_feature_labels\" fields must be equal."
    assert graph_1.edge_ids == graph_2.edge_ids, \
        "The input's and output's graph \"edge_ids\" fields must be equal."
    assert graph_1.edge_attr_labels == graph_2.edge_attr_labels, \
        "The input's and output's graph \"edge_attr_labels\" fields must be equal."
    assert graph_1.graph_id == graph_2.graph_id, \
        "The input's and output's graph \"graph_id\" fields must be equal."


########################################################################################################################
# MAIN Test ============================================================================================================
########################################################################################################################
def test_unit_add_remove_nodes():
    """
    Unit test add and remove nodes
    """

    ####################################################################################################################
    # [1.] Input graph with one node without edges =====================================================================
    ####################################################################################################################
    node_features_size = 2
    edge_features_size = 3

    node_feature_labels = ["node_feature_1", "node_feature_2"]
    edge_feature_labels = ["edge_feature_1", "edge_feature_2", "edge_feature_3"]

    node_id_1 = str(uuid.uuid4())
    node_label_1 = "label_" + str(uuid.uuid4())
    node_features_1 = np.random.randn(1, node_features_size).astype(np.float32)

    edge_index = np.empty([2, 0], dtype=int)
    edge_ids = []

    graph_id = "graph_" + str(uuid.uuid4())

    graph_1 = Data(x=torch.from_numpy(node_features_1),
                   edge_index=torch.from_numpy(edge_index),
                   edge_attr=None,
                   y=torch.from_numpy(np.array([1])),
                   node_labels=[node_label_1],
                   node_ids=[node_id_1],
                   node_feature_labels=node_feature_labels,
                   edge_ids=edge_ids,
                   edge_attr_labels=edge_feature_labels,
                   pos=None,
                   graph_id=graph_id
                   )
    print(graph_1)
    print("-----------------------------------------------------------------------------------------------------------")

    ####################################################################################################################
    # [2.] Add a second node ===========================================================================================
    ####################################################################################################################
    node_id_2 = str(uuid.uuid4())
    node_label_2 = "label_" + str(uuid.uuid4())
    node_features_2 = np.random.randn(1, node_features_size).astype(np.float32)

    graph_2 = add_node(graph_1, node_features_2, node_label_2, node_id_2)
    print(graph_2)
    print("-----------------------------------------------------------------------------------------------------------")

    # [2.1.] Check the elements that don't change ----------------------------------------------------------------------
    unchanged_fields_node_add_remove(graph_1, graph_2)

    # [2.2.] Check the elements that change - "x", "node_labels", "node_ids" -------------------------------------------
    assert torch.equal(graph_2.x, torch.vstack((graph_1.x, torch.from_numpy(node_features_2)))), \
        "The second graph's \"x\" field "
    assert graph_2.node_labels == [node_label_1, node_label_2], "The node labels of the updated graph is incorrect."
    assert graph_2.node_ids == [node_id_1, node_id_2], "The node ids of the updated graph is incorrect."

    ####################################################################################################################
    # [3.] Remove this second node =====================================================================================
    ####################################################################################################################
    nodes_nr = graph_2.x.size(dim=0)
    node_idx_to_remove = random.randint(0, nodes_nr - 1)
    graph_3 = remove_node(graph_2, node_idx_to_remove)

    # [3.1.] Check the elements that don't change ----------------------------------------------------------------------
    unchanged_fields_node_add_remove(graph_2, graph_3)

    # [3.2.] Check the elements that change - "x", "node_labels", "node_ids" -------------------------------------------
    graph_2_x = graph_2.x.cpu().detach().numpy()
    graph_3_x = graph_3.x.cpu().detach().numpy()

    assert np.array_equal(graph_3_x, np.delete(graph_2_x, node_idx_to_remove, axis=0)), \
        "The new graph's node features must have the selected node's row removed."

    graph_2_node_labels = copy.deepcopy(graph_2.node_labels)
    del graph_2_node_labels[node_idx_to_remove]
    assert graph_3.node_labels == graph_2_node_labels, "The node labels of the updated graph is incorrect."

    graph_2_node_ids = copy.deepcopy(graph_2.node_ids)
    del graph_2_node_ids[node_idx_to_remove]
    assert graph_3.node_ids == graph_2_node_ids, "The node ids of the updated graph is incorrect."
