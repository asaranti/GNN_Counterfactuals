"""
    Graph utilities: Comparison methods ...

    :author: Anna Saranti
    :copyright: © 2022 HCI-KDD (ex-AI) group
    :date: 2021-02-04
"""

import copy

import numpy as np
import torch
from torch_geometric.data.data import Data


def graphs_equal(graph_1: Data, graph_2: Data) -> bool:
    """
    Check the equality of graphs, meaning that you check each corresponding field for equality.

    :param graph_1: First graph to compare
    :param graph_2: Second graph to compare

    Returns true if all fields are equal, else returns false
    """

    graphs_equality = True

    # [1.] Check if the data fields of the two graphs are equal by checking the keys -----------------------------------
    graph_1_keys_list = graph_1.keys
    graph_2_keys_list = graph_2.keys

    if len(graph_1_keys_list) != len(graph_2_keys_list):
        print("The two graphs have different number of field elements.\n")
        print(f"Graph 1 has {len(graph_1_keys_list)} number of elements.\n")
        print(f"Graph 2 has {len(graph_2_keys_list)} number of elements.\n")
        print("The two graphs are not equal.\n")
        return False

    # [2.] Check key by key the equality -------------------------------------------------------------------------------
    if sorted(graph_1_keys_list) == sorted(graph_2_keys_list):

        for graph_key in graph_1_keys_list:

            # [2.1.] Check all fields that are obligatory in the graph -------------------------------------------------
            if graph_key == 'x':
                graphs_equality = graphs_equality and torch.equal(graph_1.x, graph_2.x)
            elif graph_key == 'edge_index':
                graphs_equality = graphs_equality and torch.equal(graph_1.edge_index, graph_2.edge_index)
            elif graph_key == 'edge_attr':
                if graph_1.edge_attr is None and graph_2.edge_attr is not None or \
                        graph_1.edge_attr is not None and graph_2.edge_attr is None:
                    print("In one graph the \"edge_attr\" is None and in the other it is not.\n")
                    print(f"Graph 1's edge_attr:{graph_1.edge_attr} number of elements.\n")
                    print(f"Graph 2's edge_attr:{graph_2.edge_attr} number of elements.\n")
                    print("The two graphs are not equal.\n")
                    return False
                if graph_1.edge_attr is not None and graph_2.edge_attr is not None:
                    graphs_equality = graphs_equality and torch.equal(graph_1.edge_attr, graph_2.edge_attr)
            elif graph_key == 'y':
                graphs_equality = graphs_equality and torch.equal(graph_1.y, graph_2.y)

            # [2.2.] Check all fields that are obligatory in the graph -------------------------------------------------
            else:
                # [2.2.1.] Check that the type of the two elements in the two graphs is the same -----------------------
                if str(type(graph_1[graph_key])) != str(type(graph_2[graph_key])):
                    print(f"The type of the fields {graph_key} differ in the two graphs.\n")
                    print(f"Graph 1's: {graph_key} has the type: {type(graph_1[graph_key])}.\n")
                    print(f"Graph 2's {graph_key} has the type: {type(graph_2[graph_key])}.\n")
                    print("The two graphs are not equal.\n")
                    return False
                # [2.2.2.] Handle each type of the two elements in the two graphs differently --------------------------
                #          Lists and arrays have to be exactly the same (not just the elements), -----------------------
                #          same goes for strings -----------------------------------------------------------------------
                else:
                    graph_key_type = str(type(graph_1[graph_key]))
                    if graph_key_type == "<class 'list'>" or graph_key_type == "<class 'str'>":
                        graphs_equality = graphs_equality and graph_1[graph_key] == graph_2[graph_key]
                    elif graph_key_type == "<class 'np.ndarray'>":
                        graphs_equality = graphs_equality and np.array_equal(graph_1[graph_key] == graph_2[graph_key])
                    else:
                        graphs_equality = False
                        print(f"The type {graph_key_type} is not allowed for further fields of the graph data "
                              f"structure.\nIt has to be either \"list\", \"np.ndarray\" or \"str\"")

    return graphs_equality


def compare_graphs_topology(graph_dataset: list):
    """
    Compare the topology of all graphs in the input dataset.
    If some of the graphs does not have the topology of the first one,

    :param graph_dataset: List with all graphs
    """

    print("Check that the graphs have the same topology ...")

    for graph_idx in range(len(graph_dataset)):

        graph = graph_dataset[graph_idx]

        if graph_idx == 0:
            graph_0 = copy.deepcopy(graph)
        else:
            b_graph_equals = torch.equal(graph_0.edge_index, graph.edge_index)
            assert b_graph_equals, "The graphs do not have the same topology. " \
                                   "The \"edge_index\" parameter is not the same."


def compare_node_features_values(graph_dataset: list):
    """
    Compare the values of the features of the nodes

    :param graph_dataset: List with all graphs
    """

    print("Check the variation of node feature values ...")
    features_shape = graph_dataset[0].x.cpu().detach().numpy().shape
    nodes_nr = features_shape[0]
    features_nr = features_shape[1]

    # [1.] Iterate over all features -----------------------------------------------------------------------------------
    for feature_idx in range(features_nr):

        print(f"Feature: {feature_idx}")

        feature_array = np.empty([nodes_nr, ])

        # [2.] Stack the feature of all nodes --------------------------------------------------------------------------
        for graph_idx in range(len(graph_dataset)):

            graph = graph_dataset[graph_idx]
            feature = graph.x.cpu().detach().numpy()[:, feature_idx]
            feature_array = np.vstack((feature_array, feature))

        # [4.] Min, max, median value of array -------------------------------------------------------------------------
        feature_min_val = np.amin(feature_array)
        feature_max_val = np.amax(feature_array)
        feature_median_val = np.median(feature_array)
        print(f"MIN val: {feature_min_val}, MAX val: {feature_max_val}, MEDIAN val: {feature_median_val}")

        # [4.] Compute the variation -----------------------------------------------------------------------------------
        feature_variation = np.var(feature_array, axis=0)

        print(f"Feature variation: {feature_variation}")
        print(f"MIN feature variation: {np.amin(feature_variation)},\n"
              f"MAX feature variation: {np.amax(feature_variation)}")
        print("-------------------------------------------------------------------------------------------------------")


def remove_duplicate_edges(edge_index_input: np.array) -> np.array:
    """
    Remove duplicate edges from the edge_index

    :param edge_index_input: Edge index input array
    :return: An array that has only unique edges
    """

    edges_pairs = list(zip(*edge_index_input))
    edges_pairs_sorted = [tuple(sorted(list(x))) for x in edges_pairs]
    edges_pairs_sorted = np.array(list(set(edges_pairs_sorted)))

    edges_pairs_sorted_array = np.transpose(edges_pairs_sorted)

    return edges_pairs_sorted_array
