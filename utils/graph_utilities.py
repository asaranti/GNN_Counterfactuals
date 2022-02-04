"""
    Graph utilities: Comparison methods ...

    :author: Anna Saranti
    :copyright: © 2022 HCI-KDD (ex-AI) group
    :date: 2021-02-04
"""

import copy

import torch


def compare_graphs_topology(graph_dataset: list):
    """
    Compare the topology of all graphs in the input dataset.
    If some of the graphs does not have the topology of the first one,

    :param graph_dataset: List with all graphs
    """

    for graph_idx in range(len(graph_dataset)):

        graph = graph_dataset[graph_idx]

        if graph_idx == 0:
            graph_0 = copy.deepcopy(graph)
        else:
            b_graph_equals = torch.equal(graph_0.edge_index, graph.edge_index)
            assert b_graph_equals, "The graphs do not have the same topology. " \
                                   "The \"edge_index\" parameter is not the same."
