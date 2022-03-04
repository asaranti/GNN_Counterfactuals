"""
    Graph utilities: Comparison methods ...

    :author: Anna Saranti
    :copyright: © 2022 HCI-KDD (ex-AI) group
    :date: 2021-02-04
"""

import copy

import numpy as np
import torch


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


