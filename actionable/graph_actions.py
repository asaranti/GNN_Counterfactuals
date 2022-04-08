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


def add_node(input_graph: torch_geometric.data.data.Data, node_features: np.array,
             label: str, node_id: str) -> torch_geometric.data.data.Data:
    """
    Add node with its features. No edges are added at this point.
    If there are already nodes with specified features, then the size of the new node's feature array
    must conform to that, since heterogeneous graphs are not allowed.

    :param input_graph: Input graph
    :param node_features: A numpy row containing the node features
    :param label: Node label, which will be added
    :param node_id: Node id, which will be added

    :return: The updated graph
    """

    # [0.] Constraint: the new node's features needs to be the same length (type) --------------------------------------
    input_graph_x = input_graph.x.cpu().detach().numpy()
    if input_graph_x is not None:
        print(input_graph_x.shape[1], node_features.shape[1])
        assert input_graph_x.shape[1] == node_features.shape[1], \
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
    output_graph_node_ids = np.append(input_graph.node_ids, [node_id], axis=0)
    output_graph_node_labels = np.append(input_graph.node_labels, [label], axis=0)

    output_graph = Data(x=torch.from_numpy(output_graph_x),
                        edge_index=input_graph.edge_index,
                        edge_attr=input_graph.edge_attr,
                        y=input_graph.y,
                        node_labels=output_graph_node_labels,
                        node_ids=output_graph_node_ids,
                        node_feature_labels=input_graph.node_feature_labels,
                        edge_ids=input_graph.edge_ids,
                        edge_attr_labels=input_graph.edge_attr_labels,
                        pos=output_pos,
                        graph_id=input_graph.graph_id,
                        dtype=torch.long)

    return output_graph


def remove_node(input_graph: torch_geometric.data.data.Data, node_index: int,
                label: str) -> torch_geometric.data.data.Data:
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
    :param label: Node label for removal

    :return: The updated graph
    """

    # [0.] Check that the index of the deleted node is valid -----------------------------------------------------------
    assert input_graph.x is not None, "No nodes saved in the graphs, the \"x\" field is None"
    assert 0 <= node_index < input_graph.num_nodes, \
        f"The index of the node {node_index} is not in accordance with the number of nodes {input_graph.num_nodes}"

    # [1.] The corresponding line needs to be removed from the "x" field of the Data class, ----------------------------
    #      corresponding to node features. -----------------------------------------------------------------------------
    input_graph_x = input_graph.x.cpu().detach().numpy()
    output_graph_x = np.delete(input_graph_x, node_index, 0)

    input_graph.node_ids = np.delete(input_graph.node_ids, node_index)

    # [2.] All edges in which the node participates need to be deleted. ------------------------------------------------
    #      The corresponding pairs in the "edge_index" need to be removed. ---------------------------------------------
    #      This part is done via the UI.
    input_graph_edge_index = input_graph.cpu().detach().edge_index.numpy()

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

            input_graph_edge_index_left_reindexed = []
            input_graph_edge_index_right_reindexed = []
            for x in input_graph_edge_index_left:
                if x > node_index:
                    input_graph_edge_index_left_reindexed.append(x - 1)
                else:
                    input_graph_edge_index_left_reindexed.append(x)
            for x in input_graph_edge_index_right:
                if x > node_index:
                    input_graph_edge_index_right_reindexed.append(x - 1)
                else:
                    input_graph_edge_index_right_reindexed.append(x)

            output_graph_edge_index = torch.from_numpy(np.row_stack((input_graph_edge_index_left_reindexed,
                                                                     input_graph_edge_index_right_reindexed)))

            # [3.] The features of all the edges that are deleted in the previous step ---------------------------------
            #          must also be deleted. ---------------------------------------------------------------------------
            if input_graph.edge_attr is not None:
                input_graph_edge_attr = input_graph.edge_attr.numpy()
                for removing_edge_index in all_indices:
                    input_graph_edge_attr = np.delete(input_graph_edge_attr, removing_edge_index, 0)
                output_graph_edge_attr = torch.from_numpy(input_graph_edge_attr)

    # [4.] Graph node labels need to be adapted ------------------------------------------------------------------------
    output_graph_node_labels = np.delete(input_graph.node_labels, np.argwhere(input_graph.node_labels == label))

    # [5.] The field "y" must not be deleted. --------------------------------------------------------------------------
    #      Nothing to do here

    # [6.] In the field position "pos" the position of the deleted node needs to be removed. ---------------------------
    output_pos = input_graph.pos

    # [7.] Output graph ------------------------------------------------------------------------------------------------
    output_graph = Data(x=torch.from_numpy(output_graph_x),
                        edge_index=output_graph_edge_index,
                        edge_attr=output_graph_edge_attr,
                        y=input_graph.y,
                        node_labels=output_graph_node_labels,
                        node_ids=input_graph.node_ids,
                        node_feature_labels=input_graph.node_feature_labels,
                        edge_ids=input_graph.edge_ids,
                        edge_attr_labels=input_graph.edge_attr_labels,
                        pos=output_pos,
                        graph_id=input_graph.graph_id,
                        dtype=torch.long
                        )

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
    assert 0 <= new_edge_index_left, \
        f"The index of the node {new_edge_index_left} is not in accordance " \
        f"with the number of nodes {input_graph.num_nodes}"
    assert 0 <= new_edge_index_right, \
        f"The index of the node {new_edge_index_right} is not in accordance " \
        f"with the number of nodes {input_graph.num_nodes}"

    # [1.] Check that the nodes specified by the input indexes are not already connected -------------------------------
    input_graph_edge_index = input_graph.edge_index.cpu().detach().numpy()
    output_graph_edge_ids = copy.deepcopy(input_graph.edge_ids)

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
        output_graph_edge_ids = np.append(output_graph_edge_ids, f"{new_edge_index_left}_{new_edge_index_right}")

        output_graph_edge_index = torch.from_numpy(np.row_stack((input_graph_edge_index_left,
                                                                 input_graph_edge_index_right)))
    else:
        output_graph_edge_index = torch.from_numpy(np.array([[new_edge_index_left],
                                                             [new_edge_index_right]]))

    # [1.] Add the node's features -------------------------------------------------------------------------------------
    if input_graph.edge_attr is not None and new_edge_attr is not None:
        input_graph_edge_attr = input_graph.edge_attr.cpu().detach().numpy()

        assert input_graph_edge_attr.shape[1] == new_edge_attr.shape[1], \
            f"The shape of the features of the new edge: {new_edge_attr.shape[1]} must conform to the shape " \
            f"of the features of the rest of the edges: {input_graph_edge_attr.shape[1]}. " \
            f"The graph must be homogeneous."
        output_graph_edge_attr = torch.from_numpy(np.row_stack((input_graph_edge_attr, new_edge_attr)))
    elif new_edge_attr is not None:
        output_graph_edge_attr = torch.from_numpy(np.array([new_edge_attr]))
    else:
        output_graph_edge_attr = None

    # [3.] In the field position "pos" the position of the deleted node needs to be removed. ---------------------------
    output_pos = input_graph.pos

    # [4.] Output graph ------------------------------------------------------------------------------------------------
    output_graph = Data(x=input_graph.x,
                        edge_index=output_graph_edge_index,
                        edge_attr=output_graph_edge_attr,
                        y=input_graph.y,
                        node_labels=input_graph.node_labels,
                        node_ids=input_graph.node_ids,
                        node_feature_labels=input_graph.node_feature_labels,
                        edge_ids=output_graph_edge_ids,
                        edge_attr_labels=input_graph.edge_attr_labels,
                        pos=output_pos,
                        graph_id=input_graph.graph_id,
                        dtype=torch.long)

    return output_graph


def remove_edge(input_graph: torch_geometric.data.data.Data, edge_index_left: int, edge_index_right: int) -> \
        torch_geometric.data.data.Data:
    """
    Remove an edge between two nodes. Check if the edge exists, remove the corresponding pair from "edge_index" and the
    attributes of this pair from the "edge_attr". All other fields stay unchanged.

    :param input_graph: Input graph
    :param edge_index_left: Index of left node of the edge
    :param edge_index_right: Index of right node of the edge

    :return: The updated graph
    """

    # [1.] Check that the nodes specified by the input indexes are not already connected -------------------------------
    output_graph_edge_index = copy.deepcopy(input_graph.edge_index)
    output_graph_edge_attr = copy.deepcopy(input_graph.edge_attr)
    output_graph_edge_ids = copy.deepcopy(input_graph.edge_ids)

    input_graph_edge_index = input_graph.edge_index.cpu().detach().numpy()

    assert input_graph_edge_index is not None, "There are no edges in the graph"

    input_graph_edge_index_left = list(input_graph_edge_index[0, :])
    input_graph_edge_index_right = list(input_graph_edge_index[1, :])

    if len(input_graph_edge_index_left) > 0 and len(input_graph_edge_index_right) > 0:
        graph_edge_pairs = list(map(lambda x: (x[0], x[1]),
                                    list(zip(input_graph_edge_index_left, input_graph_edge_index_right))))
        assert (edge_index_left, edge_index_right) in graph_edge_pairs or \
               (edge_index_right, edge_index_left) in graph_edge_pairs, \
               f"There is no edge connecting node {edge_index_left} and {edge_index_right}.\n"

        # # [2.] Remove the edge and its attributes --------------------------------------------------------------------
        if (edge_index_left, edge_index_right) in graph_edge_pairs:
            index_of_pair_to_delete = graph_edge_pairs.index((edge_index_left, edge_index_right))
        else:
            index_of_pair_to_delete = graph_edge_pairs.index((edge_index_right, edge_index_left))

        del input_graph_edge_index_left[index_of_pair_to_delete]
        del input_graph_edge_index_right[index_of_pair_to_delete]

        output_graph_edge_index = torch.from_numpy(np.row_stack((input_graph_edge_index_left,
                                                                 input_graph_edge_index_right)))
        output_graph_edge_ids = np.delete(output_graph_edge_ids, index_of_pair_to_delete)

        if input_graph.edge_attr is not None:
            input_graph_edge_attr = input_graph.edge_attr.cpu().detach().numpy()
            input_graph_edge_attr = np.delete(input_graph_edge_attr, index_of_pair_to_delete, 0)
            output_graph_edge_attr = torch.from_numpy(input_graph_edge_attr)
        else:
            output_graph_edge_attr = None

    # [3.] In the field position "pos" the position of the deleted node needs to be removed. ---------------------------
    output_pos = input_graph.pos

    # [4.] Output graph ------------------------------------------------------------------------------------------------
    output_graph = Data(x=input_graph.x,
                        edge_index=output_graph_edge_index,
                        edge_attr=output_graph_edge_attr,
                        y=input_graph.y,
                        node_labels=input_graph.node_labels,
                        node_ids=input_graph.node_ids,
                        node_feature_labels=input_graph.node_feature_labels,
                        edge_ids=output_graph_edge_ids,
                        edge_attr_labels=input_graph.edge_attr_labels,
                        pos=output_pos,
                        graph_id=input_graph.graph_id,
                        dtype=torch.long)
    return output_graph


def add_feature_all_nodes(input_graph: torch_geometric.data.data.Data, new_input_node_feature: np.array) -> \
        torch_geometric.data.data.Data:
    """
    Add a feature in all of the nodes. Basically, that means that the features field of all the nodes "x" will have
    another column. The number of rows of the input feature should be equal to the number of nodes. If the node's
    features field "x" is empty then a one column array is created. The other fields stay unchanged.

    :param input_graph: Input graph
    :param new_input_node_feature: New input feature

    :return: The updated graph
    """

    # [1.] Number of rows of the input feature should be equal to the number of nodes ----------------------------------
    nodes_nr = input_graph.num_nodes
    assert nodes_nr == new_input_node_feature.shape[0], f"The number of nodes: {nodes_nr} in the graph is not equal " \
                                                        f"to the number of rows in the input feature: " \
                                                        f"{new_input_node_feature.shape[0]}. All nodes should have " \
                                                        f"the feature, since heterogeneous graphs are not allowed."

    input_graph_x = input_graph.x.cpu().detach().numpy()
    output_graph_x = np.column_stack((input_graph_x, new_input_node_feature))

    # [2.] In the field position "pos" the position of the deleted node needs to be removed. ---------------------------
    output_pos = input_graph.pos

    # [3.] Output graph ------------------------------------------------------------------------------------------------
    output_graph = Data(x=torch.from_numpy(output_graph_x),
                        edge_index=input_graph.edge_index,
                        edge_attr=input_graph.edge_attr,
                        y=input_graph.y,
                        node_labels=input_graph.node_labels,
                        node_ids=input_graph.node_ids,
                        node_feature_labels=input_graph.node_feature_labels,
                        edge_ids=input_graph.edge_ids,
                        edge_attr_labels=input_graph.edge_attr_labels,
                        pos=output_pos,
                        graph_id=input_graph.graph_id,
                        dtype=torch.long)

    return output_graph


def remove_feature_all_nodes(input_graph: torch_geometric.data.data.Data, removed_node_feature_idx: int) -> \
        torch_geometric.data.data.Data:
    """
    Remove a feature in all of the nodes. The features field of all the nodes "x" will have one column less.
    It is presupposed that the node index is valid. If the node's features field "x" is empty then nothing is done.
    The other fields stay unchanged.

    :param input_graph: Input graph
    :param removed_node_feature_idx: Index of the column of the removed feature

    :return: The updated graph
    """

    # [0.] Check that the index of the deleted feature is valid --------------------------------------------------------
    assert input_graph.x is not None, "No nodes saved in the graphs, the \"x\" field is None"
    input_graph_x = input_graph.x.cpu().detach().numpy()
    node_features_nr = input_graph_x.shape[1]
    assert 0 <= removed_node_feature_idx < node_features_nr, \
           f"The index of the feature index: {removed_node_feature_idx} is not in accordance with the " \
           f"number of features {node_features_nr}"

    output_graph_x = np.delete(input_graph_x, removed_node_feature_idx, 1)

    # [2.] In the field position "pos" the position of the deleted node needs to be removed. ---------------------------
    output_pos = input_graph.pos

    # [3.] Output graph ------------------------------------------------------------------------------------------------
    output_graph = Data(x=torch.from_numpy(output_graph_x),
                        edge_index=input_graph.edge_index,
                        edge_attr=input_graph.edge_attr,
                        y=input_graph.y,
                        node_labels=input_graph.node_labels,
                        node_ids=input_graph.node_ids,
                        node_feature_labels=input_graph.node_feature_labels,
                        edge_ids=input_graph.edge_ids,
                        edge_attr_labels=input_graph.edge_attr_labels,
                        pos=output_pos,
                        graph_id=input_graph.graph_id,
                        dtype=torch.long)
    return output_graph


def add_feature_all_edges(input_graph: torch_geometric.data.data.Data, new_input_edge_feature: np.array) -> \
        torch_geometric.data.data.Data:
    """
    Add a feature in all of the edges. Basically, that means that the features field of all the edges "edge_attr"
    will have another column. The number of rows of the input attribute should be equal to the number of edges.
    If the edges's attributes field "edge_attr" is empty or None then a one column array is created.
    The other fields stay unchanged.
    """

    # [1.] Number of rows of the input feature should be equal to the number of nodes ----------------------------------
    edges_nr = input_graph.num_edges
    assert edges_nr == new_input_edge_feature.shape[0], f"The number of edges: {edges_nr} in the graph is not equal " \
                                                        f"to the number of rows in the input attribute: " \
                                                        f"{new_input_edge_feature.shape[0]}. All edges should have " \
                                                        f"the attribute, since heterogeneous graphs are not allowed."

    if input_graph.edge_attr is not None:
        input_graph_edge_attr = input_graph.edge_attr.cpu().detach().numpy()
        output_graph_edge_attr = torch.from_numpy(np.column_stack((input_graph_edge_attr, new_input_edge_feature)))
    else:
        output_graph_edge_attr = torch.from_numpy([new_input_edge_feature])

    # [2.] In the field position "pos" the position of the deleted node needs to be removed. ---------------------------
    output_pos = input_graph.pos

    # [3.] Output graph ------------------------------------------------------------------------------------------------
    output_graph = Data(x=input_graph.x,
                        edge_index=input_graph.edge_index,
                        edge_attr=output_graph_edge_attr,
                        y=input_graph.y,
                        node_labels=input_graph.node_labels,
                        node_ids=input_graph.node_ids,
                        node_feature_labels=input_graph.node_feature_labels,
                        edge_ids=input_graph.edge_ids,
                        edge_attr_labels=input_graph.edge_attr_labels,
                        pos=output_pos,
                        graph_id=input_graph.graph_id,
                        dtype=torch.long)

    return output_graph


def remove_feature_all_edges(input_graph: torch_geometric.data.data.Data, removed_edge_attribute_idx: int) -> \
        torch_geometric.data.data.Data:
    """
    Remove a feature in all of the edges. The attributes field of all the edges "edge_attr" will have one column less.
    It is presupposed that the edge index is valid. If the edge's features field "edge_attr" is empty then nothing
    is done. The other fields stay unchanged.

    :param input_graph: Input graph
    :param removed_edge_attribute_idx: Index of the column of the removed attribute

    :return: The updated graph
    """

    # [1.] Check that the index of the deleted feature is valid --------------------------------------------------------
    if input_graph.edge_attr is not None:

        input_graph_edge_attr = input_graph.edge_attr.cpu().detach().numpy()

        edge_attributes_nr = input_graph_edge_attr.shape[1]
        assert 0 <= removed_edge_attribute_idx < edge_attributes_nr, \
            f"The index of the feature index: {removed_edge_attribute_idx} is not in accordance with the " \
            f"number of features {edge_attributes_nr}"

        output_graph_edge_attr = torch.from_numpy(np.delete(input_graph_edge_attr, removed_edge_attribute_idx, 1))
    else:
        output_graph_edge_attr = None

    # [2.] In the field position "pos" the position of the deleted node needs to be removed. ---------------------------
    output_pos = input_graph.pos

    # [3.] Output graph ------------------------------------------------------------------------------------------------
    output_graph = Data(x=input_graph.x,
                        edge_index=input_graph.edge_index,
                        edge_attr=output_graph_edge_attr,
                        y=input_graph.y,
                        node_labels=input_graph.node_labels,
                        node_ids=input_graph.node_ids,
                        node_feature_labels=input_graph.node_feature_labels,
                        edge_ids=input_graph.edge_ids,
                        edge_attr_labels=input_graph.edge_attr_labels,
                        pos=output_pos,
                        graph_id=input_graph.graph_id,
                        dtype=torch.long)

    return output_graph
