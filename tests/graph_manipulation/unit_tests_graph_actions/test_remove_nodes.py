"""
    Test remove node(s)

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-04-12
"""

import os
import pickle
import random
import torch

import numpy as np
import torch_geometric

from actionable.graph_actions import remove_node


def check_node_remove(input_graph: torch_geometric.data.data.Data,
                      output_graph: torch_geometric.data.data.Data,
                      node_index: int):
    """
    Check the node removal

    :param input_graph: Input graph before node addition
    :param output_graph: Output graph after node addition
    :param node_index: Node index, which will be removed - It is its numeric index, has nothing to do with its ID
    """

    # [1.] Field "x" of the output_graph will have an removed column ---------------------------------------------------
    input_graph_x = input_graph.x.cpu().detach().numpy()
    output_graph_x = output_graph.x.cpu().detach().numpy()

    assert np.array_equal(np.delete(input_graph_x, node_index, 0), output_graph_x), \
        f"The output graph's \"x\" field must have the corresponding column with index: {node_index} removed."

    # [2.] "y", "pos", "graph_id", "node_feature_labels", "edge_attr_labels" stay intact - don't change ----------------
    assert torch.equal(input_graph.y, output_graph.y), "The input's and output's graph \"y\" fields must be equal."
    assert input_graph.pos == output_graph.pos, "The input's and output's graph \"pos\" fields must be equal."
    assert input_graph.graph_id == output_graph.graph_id, "The input's and output's graph \"graph_id\" fields must " \
                                                          "be equal."
    assert input_graph.node_feature_labels == output_graph.node_feature_labels, \
        "The input's and output's graph \"node_feature_labels\" fields must be equal."
    assert input_graph.edge_attr_labels == output_graph.edge_attr_labels, \
        "The input's and output's graph \"edge_attr_labels\" fields must be equal."

    # [3.]  "edge_index", "edge_attr", "edge_ids" will probably change -------------------------------------------------
    # [3.1.] "edge_index" ----------------------------------------------------------------------------------------------
    input_graph_edge_index = input_graph.edge_index.cpu().detach().numpy()
    output_graph_edge_index = output_graph.edge_index.cpu().detach().numpy()

    if input_graph.edge_index is not None:

        input_graph_edge_index_left = list(input_graph.edge_index[0, :])
        input_graph_edge_index_right = list(input_graph.edge_index[1, :])

        if len(input_graph_edge_index_left) > 0 and len(input_graph_edge_index_right) > 0:

            left_indices = [i for i, x in enumerate(input_graph_edge_index_left) if x == (node_index - 1)]
            right_indices = [i for i, x in enumerate(input_graph_edge_index_right) if x == (node_index - 1)]

            all_indices = list(set(left_indices + right_indices))
            input_graph_edge_index_after_delete = np.delete(input_graph_edge_index, all_indices, axis=1)
            input_graph_edge_index_after_delete[input_graph_edge_index_after_delete > node_index] -= 1

            assert np.array_equal(input_graph_edge_index_after_delete,
                                  output_graph_edge_index), "For the deleted edges the \"edge_index\" elements that " \
                                                            "contained the index of the node must be removed. " \
                                                            "Reindexing -1 must be done."

            # [3.2.] "edge_attr" ---------------------------------------------------------------------------------------
            if input_graph.edge_attr is not None:

                input_graph_edge_attr = input_graph.edge_attr.cpu().detach().numpy()
                output_graph_edge_attr = output_graph.edge_attr.cpu().detach().numpy()

                assert np.array_equal(np.delete(input_graph_edge_attr, all_indices, axis=0),
                                      output_graph_edge_attr), \
                    "For the deleted edges the \"edge_attr\" elements that contained the index of the node must be " \
                    "removed."

            # [3.3.] "edge_ids" ----------------------------------------------------------------------------------------
            output_graph_edge_ids = output_graph.edge_ids
            assert np.array_equal(list(np.delete(input_graph.edge_ids, all_indices, axis=0)),
                                  output_graph_edge_ids), \
                "For the deleted edges the \"edge_ids\" elements that contained the index of the node must be " \
                "removed."

    # [4.] "node_ids" and "node_labels" have one element less ----------------------------------------------------------
    assert input_graph.node_ids[:node_index] + input_graph.node_ids[node_index + 1:] == output_graph.node_ids, \
        "The input's and output's graph \"node_ids\" must have the node_id at the node_index removed."

    assert input_graph.node_labels[:node_index] + input_graph.node_labels[node_index + 1:] == \
           output_graph.node_labels, "The input's and output's graph \"node_labels\" must have the \"node_label\" " \
                                     "at the \"node_index\" removed."


########################################################################################################################
# MAIN Test ============================================================================================================
########################################################################################################################
def test_unit_remove_nodes():
    """
    Unit test remove nodes
    """

    # [1.] Transformation Experiment ::: From PPI to Pytorch_Graph -----------------------------------------------------
    dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
    dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))

    dataset_len = len(dataset)
    graph_idx = random.randint(0, dataset_len)
    input_graph = dataset[graph_idx]

    # [2.] Try node addition(s) ----------------------------------------------------------------------------------------
    node_removals = 10
    for node_removal in range(node_removals):

        nodes_nr = input_graph.x.size(dim=0)
        node_index = random.randint(0, nodes_nr)
        output_graph = remove_node(input_graph, node_index)

        # [3.] Check that the node addition is successful --------------------------------------------------------------
        check_node_remove(input_graph, output_graph, node_index)

        # [4.] Copy and repeat -----------------------------------------------------------------------------------------
        input_graph = output_graph
