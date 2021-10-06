"""
    All possible graph actions

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-10-05
"""

import copy

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data


def add_node(input_graph: torch_geometric.data.data.Data, node_features: np.array) -> torch_geometric.data.data.Data:
    """
    Add node with its features. No edges are added at this point.
    If there are already nodes with specified features, then the size of the new node's feature array
    must conform to that, since heterogeneous graphs are not allowed.

    :param input_graph: Input graph
    :param node_features: A numpy row containing the node features

    :return: The updated graph
    """

    # [0.] Constraint: the new node's features needs to be the same length (type) --------------------------------------
    input_graph_x = input_graph.x.numpy()
    if input_graph_x is not None:
        assert input_graph_x.shape[1] != node_features.shape[1], \
            "The shape of the features of the new node must conform to the shape " \
            "of the features of the rest of the nodes. The graph must be homogeneous."

    # [1.] Add the node's features -------------------------------------------------------------------------------------
    if input_graph_x is not None:
        output_graph_x = np.row_stack((input_graph_x,
                                       node_features))
    else:
        output_graph_x = np.array([node_features])

    # [2.] If pos is not None, then it needs to contain a new pos ------------------------------------------------------
    #      TODO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    output_pos = input_graph.pos

    # [3.] Output graph ------------------------------------------------------------------------------------------------
    output_graph = Data(x=torch.from_numpy(output_graph_x),
                        edge_index=input_graph.edge_index,
                        edge_attr=input_graph.edge_attr,
                        y=input_graph.y,
                        pos=output_pos,
                        dtype=torch.long)
    return output_graph


def remove_node(input_graph: torch_geometric.data.data.Data, node_index: int) -> torch_geometric.data.data.Data:
    """
    Remove one node from the graph according to its index.
    It is presupposed that the node index is valid.
    For example, one cannot delete the last node of a graph or a node with a higher index than the one allowed by the
    number of nodes.

    The following actions must be taken:
    [1.] The corresponding line needs to be removed from the "x" field of the Data class,
         corresponding to node features.
    [2.] All edges in which the node participates need to be deleted.
         The corresponding pairs in the "edge_index" need to be removed.
    [3.] The features of all the edges that are deleted in the previous step
         must also be deleted.
    [4.] The field "y" must not be deleted.
    [5.] In the field position "pos" the position of the deleted node needs to be removed.

    Beware that all those fields are Optional and by default None according to the specification.

    :param input_graph: Input graph
    :param node_index: Node index for removal

    :return: The updated graph
    """

    # [0.] Make a deepcopy of the input graph --------------------------------------------------------------------------
    #      and check that the index of the deleted node is valid -------------------------------------------------------
    assert input_graph.x is not None, "No nodes saved in the graphs, the \"x\" field is None"
    assert 0 <= node_index < input_graph.num_nodes, \
        f"The index of the node {node_index} is not in accordance with the number of nodes {input_graph.num_nodes}"

    # [1.] The corresponding line needs to be removed from the "x" field of the Data class, ----------------------------
    #      corresponding to node features. -----------------------------------------------------------------------------
    input_graph_x = input_graph.x.numpy()
    output_graph_x = np.delete(input_graph_x, node_index, 0)

    # [2.] All edges in which the node participates need to be deleted. ------------------------------------------------
    #      The corresponding pairs in the "edge_index" need to be removed. ---------------------------------------------
    input_graph_edge_index = input_graph.edge_index.numpy()

    output_graph_edge_index = copy.deepcopy(input_graph.edge_index)
    output_graph_edge_attr = copy.deepcopy(input_graph.edge_attr)

    if input_graph_edge_index is not None:

        input_graph_edge_index_left = list(input_graph_edge_index[0, :])
        input_graph_edge_index_right = list(input_graph_edge_index[1, :])

        if len(input_graph_edge_index_left) > 0 and len(input_graph_edge_index_right) > 0:

            left_indices = [i for i, x in enumerate(input_graph_edge_index_left) if x == (node_index - 1)]
            right_indices = [i for i, x in enumerate(input_graph_edge_index_right) if x == (node_index - 1)]

            all_indices = list(set(left_indices + right_indices))
            all_indices.sort()
            all_indices.reverse()

            for removing_edge_index in all_indices:
                del input_graph_edge_index_left[removing_edge_index]
                del input_graph_edge_index_right[removing_edge_index]

            output_graph_edge_index = torch.from_numpy(np.row_stack((input_graph_edge_index_left,
                                                                     input_graph_edge_index_right)))

            # [3.] The features of all the edges that are deleted in the previous step ---------------------------------
            #          must also be deleted. ---------------------------------------------------------------------------
            input_graph_edge_attr = input_graph.edge_attr.numpy()
            for removing_edge_index in all_indices:
                input_graph_edge_attr = np.delete(input_graph_edge_attr, removing_edge_index, 0)
            output_graph_edge_attr = torch.from_numpy(input_graph_edge_attr)

    # [4.] The field "y" must not be deleted. --------------------------------------------------------------------------
    #      Nothing to do here

    # [5.] In the field position "pos" the position of the deleted node needs to be removed. ---------------------------
    output_pos = input_graph.pos

    # [6.] Output graph ------------------------------------------------------------------------------------------------
    output_graph = Data(x=torch.from_numpy(output_graph_x),
                        edge_index=output_graph_edge_index,
                        edge_attr=output_graph_edge_attr,
                        y=input_graph.y,
                        pos=output_pos,
                        dtype=torch.long)
    return output_graph


def add_edge(input_graph: torch_geometric.data.data.Data, new_edge_index_left: int, new_edge_index_right: int,
             new_edge_attr: np.array) -> \
        torch_geometric.data.data.Data:
    """
    Add edge to the graph. The "left" and "right" nodes (imaginary in undirected graphs) must exist.
    Since we are not dealing with multi-graphs, it must be checked that the nodes specified by the input indexes
    are not already connected. The new edge has to have the same feature type as the other edges in the network,
    if they have any. The fields "pos" specifying the position of the nodes and "y" denoting the class remain unchanged.

    :param input_graph: Input graph
    :param new_edge_index_left: Index of left node of new edge
    :param new_edge_index_right: Index of right node of new edge
    :param new_edge_attr: Attribute(s) of new edge

    :return: The updated graph
    """

    # [0.] Check that nodes exist in the graph and check that the index of the left and right node is valid ------------
    assert input_graph.x is not None, "No nodes saved in the graphs, the \"x\" field is None"
    assert 0 <= new_edge_index_left < input_graph.num_nodes, \
        f"The index of the node {new_edge_index_left} is not in accordance " \
        f"with the number of nodes {input_graph.num_nodes}"
    assert 0 <= new_edge_index_right < input_graph.num_nodes, \
        f"The index of the node {new_edge_index_right} is not in accordance " \
        f"with the number of nodes {input_graph.num_nodes}"

    # [1.] Check that the nodes specified by the input indexes are not already connected -------------------------------
    output_graph_edge_index = copy.deepcopy(input_graph.edge_index)
    input_graph_edge_index = input_graph.edge_index.numpy()

    if input_graph_edge_index is not None:

        input_graph_edge_index_left = list(input_graph_edge_index[0, :])
        input_graph_edge_index_right = list(input_graph_edge_index[1, :])

        if len(input_graph_edge_index_left) > 0 and len(input_graph_edge_index_right) > 0:
            graph_edge_pairs = list(map(lambda x: (x[0], x[1]),
                                        list(zip(input_graph_edge_index_left, input_graph_edge_index_right))))
            assert (new_edge_index_left, new_edge_index_right) not in graph_edge_pairs and \
                (new_edge_index_right, new_edge_index_left) not in graph_edge_pairs, \
                f"There is already an edge connecting node {new_edge_index_left} and {new_edge_index_right}.\n" \
                f"Multi-graphs are not allowed."

        # Add edge in the graph edge index ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        input_graph_edge_index_left.append(new_edge_index_left)
        input_graph_edge_index_right.append(new_edge_index_right)

        output_graph_edge_index = torch.from_numpy(np.row_stack((input_graph_edge_index_left,
                                                                 input_graph_edge_index_right)))

    # [1.] Add the node's features -------------------------------------------------------------------------------------
    input_graph_edge_attr = input_graph.edge_attr.numpy()
    if input_graph_edge_attr is not None:
        assert input_graph_edge_attr.shape[1] != input_graph_edge_attr.shape[1], \
            "The shape of the features of the new edge must conform to the shape " \
            "of the features of the rest of the edges. The graph must be homogeneous."
        output_graph_edge_attr = np.row_stack((input_graph_edge_attr, new_edge_attr))
    else:
        output_graph_edge_attr = np.array([new_edge_attr])

    # [3.] In the field position "pos" the position of the deleted node needs to be removed. ---------------------------
    output_pos = input_graph.pos

    # [4.] Output graph ------------------------------------------------------------------------------------------------
    output_graph = Data(x=input_graph.x,
                        edge_index=output_graph_edge_index,
                        edge_attr=output_graph_edge_attr,
                        y=input_graph.y,
                        pos=output_pos,
                        dtype=torch.long)
    return output_graph

# def remove_edge(input_graph: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data:

# def add_feature_all_nodes(input_graph: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data:

# def remove_feature_all_nodes(input_graph: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data:

# def add_feature_all_edges(input_graph: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data:

# def remove_feature_all_edges(input_graph: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data:
