"""
    Utilities for tests

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-04-14
"""

import numpy as np

import torch
import torch_geometric
from torch_geometric.data import Data


def unchanged_fields_node_add(graph_1: Data, graph_2: Data):
    """
    Unchanged fields for node add

    :param graph_1: Input graph before node addition
    :param graph_2: Output graph after node addition
    """

    # "edge_index", "edge_attr", "y", "edge_ids", "edge_attr_labels", "pos", "graph_id", "node_feature_labels" ---------
    # stay intact - don't change ---------------------------------------------------------------------------------------

    # TODO: Check if "edge_index" is principally allowed to be None ----------------------------------------------------
    assert torch.equal(graph_1.edge_index, graph_2.edge_index), \
        "The input's and output's graph \"edge_index\" fields must be equal."

    if graph_1.edge_attr is None:
        assert graph_2.edge_attr is None, "If the input graph's \"edge_attr\" is None, then after a node " \
                                               "addition should keep the \"edge_attr\" as None."
    else:
        assert torch.equal(graph_1.edge_attr, graph_2.edge_attr), \
            "The input's and output's graph \"edge_attr\" fields must be equal."
    assert torch.equal(graph_1.y, graph_2.y), "The input's and output's graph \"y\" fields must be equal."
    assert graph_1.edge_ids == graph_2.edge_ids, "The input's and output's graph \"edge_ids\" fields must be equal."
    assert graph_1.edge_attr_labels == graph_2.edge_attr_labels, \
        "The input's and output's graph \"edge_attr_labels\" fields must be equal."
    assert graph_1.pos == graph_2.pos, "The input's and output's graph \"pos\" fields must be equal."
    assert graph_1.graph_id == graph_2.graph_id, "The input's and output's graph \"graph_id\" fields must be equal."
    assert graph_1.node_feature_labels == graph_2.node_feature_labels, \
        "The input's and output's graph \"node_feature_labels\" fields must be equal."


def unchanged_fields_node_remove(graph_1: Data, graph_2: Data):
    """
    Unchanged fields for node removal

    :param graph_1: Input graph before node addition
    :param graph_2: Output graph after node addition
    """

    assert torch.equal(graph_1.y, graph_2.y), "The input's and output's graph \"y\" fields must be equal."
    assert graph_1.pos == graph_2.pos, "The input's and output's graph \"pos\" fields must be equal."
    assert graph_1.graph_id == graph_2.graph_id, "The input's and output's graph \"graph_id\" fields must " \
                                                 "be equal."
    assert graph_1.node_feature_labels == graph_2.node_feature_labels, \
        "The input's and output's graph \"node_feature_labels\" fields must be equal."
    assert graph_1.edge_attr_labels == graph_2.edge_attr_labels, \
        "The input's and output's graph \"edge_attr_labels\" fields must be equal."


def unchanged_fields_node_add_remove_without_edges(graph_1: Data, graph_2: Data):
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


def unchanged_fields_edge_add_remove(graph_1: Data, graph_2: Data):
    """
    Unchanged fields for edge add and remove
    """

    assert torch.equal(graph_1.x, graph_2.x), "The input's and output's graph \"x\" fields must be equal."
    assert torch.equal(graph_1.y, graph_2.y), "The input's and output's graph \"y\" fields must be equal."
    assert graph_1.node_labels == graph_2.node_labels, \
        "The input's and output's graph \"node_labels\" fields must be equal."
    assert graph_1.node_ids == graph_2.node_ids, \
        "The input's and output's graph \"node_ids\" fields must be equal."
    assert graph_1.node_feature_labels == graph_2.node_feature_labels, \
        "The input's and output's graph \"node_feature_labels\" fields must be equal."
    assert graph_1.edge_attr_labels == graph_2.edge_attr_labels, \
        "The input's and output's graph \"edge_attr_labels\" fields must be equal."
    assert graph_1.pos == graph_2.pos, "The input's and output's graph \"pos\" fields must be equal."
    assert graph_1.graph_id == graph_2.graph_id, "The input's and output's graph \"graph_id\" fields must be equal."


def unchanged_fields_feature_add(graph_1: Data, graph_2: Data):
    """
    Unchanged fields for feature add

    :param graph_1: Input graph before feature addition
    :param graph_2: Output graph after feature addition
    """

    # "edge_index", "edge_attr", "y", "node_labels", "node_ids", "edge_ids", "edge_attr_labels", "pos", "graph_id" -----
    # stay intact - don't change ---------------------------------------------------------------------------------------

    # TODO: Check if "edge_index" is principally allowed to be None ----------------------------------------------------
    assert torch.equal(graph_1.edge_index, graph_2.edge_index), \
        "The input's and output's graph \"edge_index\" fields must be equal."

    if graph_1.edge_attr is None:
        assert graph_2.edge_attr is None, "If the input graph's \"edge_attr\" is None, then after a node " \
                                          "addition should keep the \"edge_attr\" as None."
    else:
        assert torch.equal(graph_1.edge_attr, graph_2.edge_attr), \
            "The input's and output's graph \"edge_attr\" fields must be equal."
    assert graph_1.node_labels == graph_2.node_labels, \
        "The input's and output's graph \"node_labels\" fields must be equal."
    assert graph_1.node_ids == graph_2.node_ids, \
        "The input's and output's graph \"node_ids\" fields must be equal."
    assert torch.equal(graph_1.y, graph_2.y), "The input's and output's graph \"y\" fields must be equal."
    assert graph_1.edge_ids == graph_2.edge_ids, "The input's and output's graph \"edge_ids\" fields must be equal."
    assert graph_1.edge_attr_labels == graph_2.edge_attr_labels, \
        "The input's and output's graph \"edge_attr_labels\" fields must be equal."
    assert graph_1.pos == graph_2.pos, "The input's and output's graph \"pos\" fields must be equal."
    assert graph_1.graph_id == graph_2.graph_id, "The input's and output's graph \"graph_id\" fields must be equal."



########################################################################################################################
def check_edge_removal_after_node_remove(graph_1: torch_geometric.data.data.Data,
                                         graph_2: torch_geometric.data.data.Data,
                                         node_idx_to_remove: int):
    """
    Make the checks that need (or not need) to be made on the edges, after some nodes are removed

    :param graph_1: Input graph before node addition
    :param graph_2: Output graph after node addition
    :param node_idx_to_remove: Node index, which will be removed -
                               It is its numeric index, has nothing to do with its ID
    """

    input_graph_edge_index = graph_1.edge_index.cpu().detach().numpy()
    output_graph_edge_index = graph_2.edge_index.cpu().detach().numpy()

    if graph_1.edge_index is not None:

        input_graph_edge_index_left = list(graph_1.edge_index[0, :])
        input_graph_edge_index_right = list(graph_1.edge_index[1, :])

        if len(input_graph_edge_index_left) > 0 and len(input_graph_edge_index_right) > 0:

            left_indices = [i for i, x in enumerate(input_graph_edge_index_left) if x == node_idx_to_remove]
            right_indices = [i for i, x in enumerate(input_graph_edge_index_right) if x == node_idx_to_remove]

            all_indices = list(set(left_indices + right_indices))
            input_graph_edge_index_after_delete = np.delete(input_graph_edge_index, all_indices, axis=1)
            input_graph_edge_index_after_delete[input_graph_edge_index_after_delete > node_idx_to_remove] -= 1

            assert np.array_equal(input_graph_edge_index_after_delete,
                                  output_graph_edge_index), "For the deleted edges the \"edge_index\" elements that " \
                                                            "contained the index of the node must be removed. " \
                                                            "Reindexing -1 must be done."

            # [3.2.] "edge_attr" ---------------------------------------------------------------------------------------
            if graph_1.edge_attr is not None:
                input_graph_edge_attr = graph_1.edge_attr.cpu().detach().numpy()
                output_graph_edge_attr = graph_2.edge_attr.cpu().detach().numpy()

                assert np.array_equal(np.delete(input_graph_edge_attr, all_indices, axis=0),
                                      output_graph_edge_attr), \
                    "For the deleted edges the \"edge_attr\" elements that contained the index of the node must be " \
                    "removed."

            # [3.3.] "edge_ids" ----------------------------------------------------------------------------------------
            output_graph_edge_ids = graph_2.edge_ids
            assert np.array_equal(list(np.delete(graph_1.edge_ids, all_indices, axis=0)),
                                  output_graph_edge_ids), \
                "For the deleted edges the \"edge_ids\" elements that contained the index of the node must be " \
                "removed."

