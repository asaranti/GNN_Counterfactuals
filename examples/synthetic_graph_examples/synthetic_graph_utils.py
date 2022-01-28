"""
    Synthetic graph utils methods

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-12-17
"""

import numpy as np


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

