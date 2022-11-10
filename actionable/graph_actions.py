"""
    All possible graph actions

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-10-05
"""

import copy
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

from constraints.graph_constraints import check_data_format_consistency
from utils.dataset_save import append_action_dataset_history


def add_node(input_graph: torch_geometric.data.data.Data,
             node_features: np.array,
             label: str,
             node_id: str,
             dataset_name: str = None,
             user_token: str = None,
             b_save_actions_history: bool = False) -> torch_geometric.data.data.Data:
    """
    Add node with its features. No edges are added at this point.
    If there are already nodes with specified features, then the size of the new node's feature array
    must conform to that, since heterogeneous graphs are not allowed.
    If not in dataset reconstruction phase, save the actions history.

    :param input_graph: Input graph
    :param node_features: A numpy row containing the node features
    :param label: Node label, which will be added
    :param node_id: Node id, which will be added
    :param dataset_name: Dataset name that specifies the name of the model too
    :param user_token: The user token that defines the sub-folder where the actions will be stored to
    :param b_save_actions_history: For b_save_actions_history==False is used for the graph reconstruction from the file.

    :return: The updated graph
    """

    ####################################################################################################################
    # [0.] Constraints/Requirements ====================================================================================
    ####################################################################################################################
    # [0.1.] Check the types of the input graph's object ---------------------------------------------------------------
    check_data_format_consistency(input_graph)

    # [0.2.] Check the type of node features ---------------------------------------------------------------------------
    assert node_features.dtype == np.float32, f"The type of the node features must be: \"np.float32\".\n" \
                                              f"Instead it is: {node_features.dtype}"

    # [0.3.] Constraint: the new node's features needs to be the same length (type) ------------------------------------
    input_graph_x = input_graph.x.cpu().detach().numpy()
    if input_graph_x is not None:
        print(input_graph_x.shape[1], node_features.shape[1])
        assert input_graph_x.shape[1] == node_features.shape[1], \
            "The shape of the features of the new node must conform to the shape " \
            "of the features of the rest of the nodes. The graph must be homogeneous."

    ####################################################################################################################
    # [1.] Add the node's features =====================================================================================
    ####################################################################################################################
    if input_graph_x is not None:
        output_graph_x = np.row_stack((input_graph_x,
                                       node_features))
    else:
        output_graph_x = np.array([node_features])

    ####################################################################################################################
    # [2.] If pos is not None, then it needs to contain a new pos ======================================================
    #      TODO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ####################################################################################################################
    output_pos = input_graph.pos

    ####################################################################################################################
    # [3.] Take over the attribute labels of the edges, if they exist --------------------------------------------------
    ####################################################################################################################
    if hasattr(input_graph, 'edge_attr_labels'):
        output_graph_edge_attr_labels = input_graph.edge_attr_labels
    else:
        output_graph_edge_attr_labels = None

    ####################################################################################################################
    # [4.] Output graph ================================================================================================
    ####################################################################################################################
    output_graph_node_ids = copy.deepcopy(input_graph.node_ids)
    output_graph_node_labels = copy.deepcopy(input_graph.node_labels)
    output_graph_node_ids.append(node_id)
    output_graph_node_labels.append(label)

    output_graph = Data(x=torch.from_numpy(output_graph_x),
                        edge_index=input_graph.edge_index,
                        edge_attr=input_graph.edge_attr,
                        y=input_graph.y,
                        node_labels=output_graph_node_labels,
                        node_ids=output_graph_node_ids,
                        node_feature_labels=input_graph.node_feature_labels,
                        edge_ids=input_graph.edge_ids,
                        edge_attr_labels=output_graph_edge_attr_labels,
                        pos=output_pos,
                        graph_id=input_graph.graph_id
                        )

    check_data_format_consistency(output_graph)

    # [5.] Save/append the action in the local file if "b_save_actions_history" ----------------------------------------
    #      If in graph reconstruction phase b_save_actions_history == False, then there is no need to save -------------
    if b_save_actions_history:
        date_time = datetime.utcnow()
        str_date_time = date_time.strftime("%Y%m%d_%H%M%S")
        action_description = f"graph_id:{input_graph.graph_id}," \
                             f"add_node," \
                             f"node_id:{node_id}," \
                             f"label:{label}," \
                             f"node_features:{np.array2string(node_features, separator=' ')}," \
                             f"{str_date_time}" \
                             f"\n"
        append_action_dataset_history(dataset_name, user_token, action_description)

    return output_graph


def remove_node(input_graph: torch_geometric.data.data.Data,
                node_index: int,
                dataset_name: str = None,
                user_token: str = None,
                b_save_actions_history: bool = False) -> torch_geometric.data.data.Data:
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
    [6.] If not in dataset reconstruction phase, save the actions history.

    Beware that all those fields are Optional and by default None according to the specification.

    :param input_graph: Input graph
    :param node_index: Node index for removal
    :param dataset_name: Dataset name that specifies the name of the model too
    :param user_token: The user token that defines the sub-folder where the actions will be stored to
    :param b_save_actions_history: If True save the actions history in the dedicated file, else don't save the action.
    For b_save_actions_history==False is used for the graph reconstruction from the file.

    :return: The updated graph
    """

    ####################################################################################################################
    # [0.] Constraints/Requirements ====================================================================================
    ####################################################################################################################
    # [0.1.] Check the types of the input graph's object ---------------------------------------------------------------
    print("Start consistency check")
    check_data_format_consistency(input_graph)

    # [0.2.] Check that the index of the deleted node is valid ---------------------------------------------------------
    assert input_graph.x is not None, "No nodes saved in the graphs, the \"x\" field is None"
    assert 0 <= node_index < input_graph.num_nodes, \
        f"The index of the node {node_index} is not in accordance with the number of nodes {input_graph.num_nodes}"

    # [1.] The corresponding line needs to be removed from the "x" field of the Data class, ----------------------------
    #      corresponding to node features. -----------------------------------------------------------------------------
    input_graph_x = input_graph.x.cpu().detach().numpy()
    output_graph_x = np.delete(input_graph_x, node_index, 0)

    output_graph_node_ids = copy.deepcopy(input_graph.node_ids)
    del output_graph_node_ids[node_index]

    # [2.] All edges in which the node participates need to be deleted. ------------------------------------------------
    #      The corresponding pairs in the "edge_index" need to be removed. ---------------------------------------------
    #      This part is also done via the UI. --------------------------------------------------------------------------
    input_graph_edge_index = input_graph.edge_index.cpu().detach().numpy()

    output_graph_edge_index = copy.deepcopy(input_graph.edge_index)
    output_graph_edge_attr = copy.deepcopy(input_graph.edge_attr)
    output_graph_edge_ids = copy.deepcopy(input_graph.edge_ids)

    if input_graph_edge_index is not None:

        input_graph_edge_index_left = list(input_graph_edge_index[0, :])
        input_graph_edge_index_right = list(input_graph_edge_index[1, :])

        if len(input_graph_edge_index_left) > 0 and len(input_graph_edge_index_right) > 0:

            left_indices = [i for i, x in enumerate(input_graph_edge_index_left) if x == node_index]
            right_indices = [i for i, x in enumerate(input_graph_edge_index_right) if x == node_index]

            all_indices = list(set(left_indices + right_indices))
            all_indices.sort()
            all_indices.reverse()

            for removing_edge_index in all_indices:
                del input_graph_edge_index_left[removing_edge_index]
                del input_graph_edge_index_right[removing_edge_index]
                del output_graph_edge_ids[removing_edge_index]

            # [3.] Move the indexes of the nodes that have index > of the deleted node by (-1) -------------------------
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

            # [4.] The features of all the edges that are deleted in the previous step ---------------------------------
            #      must also be deleted. -------------------------------------------------------------------------------
            if input_graph.edge_attr is not None:
                input_graph_edge_attr = input_graph.edge_attr.numpy()
                for removing_edge_index in all_indices:
                    input_graph_edge_attr = np.delete(input_graph_edge_attr, removing_edge_index, 0)
                output_graph_edge_attr = torch.from_numpy(input_graph_edge_attr)

    # [5.] Graph node labels need to be adapted ------------------------------------------------------------------------
    output_graph_node_labels = copy.deepcopy(input_graph.node_labels)
    del output_graph_node_labels[node_index]

    # [6.] The field "y" must not be deleted. --------------------------------------------------------------------------
    #      Nothing to do here

    # [7.] In the field position "pos" the position of the deleted node needs to be removed. ---------------------------
    output_pos = input_graph.pos

    # [8.] Take over the attribute labels of the edges, if they exist --------------------------------------------------
    if hasattr(input_graph, 'edge_attr_labels'):
        output_graph_edge_attr_labels = input_graph.edge_attr_labels
    else:
        output_graph_edge_attr_labels = None

    # [9.] Output graph ------------------------------------------------------------------------------------------------
    output_graph = Data(x=torch.from_numpy(output_graph_x),
                        edge_index=output_graph_edge_index,
                        edge_attr=output_graph_edge_attr,
                        y=input_graph.y,
                        node_labels=output_graph_node_labels,
                        node_ids=output_graph_node_ids,
                        node_feature_labels=input_graph.node_feature_labels,
                        edge_ids=output_graph_edge_ids,
                        edge_attr_labels=output_graph_edge_attr_labels,
                        pos=output_pos,
                        graph_id=input_graph.graph_id,
                        )

    # [10.] Make the last consistency check before saving the actions and returning the graph --------------------------
    print("End consistency check")
    check_data_format_consistency(output_graph)

    # [11.] Save/append the action in the local file if "b_save_actions_history" ---------------------------------------
    #       If in graph reconstruction phase b_save_actions_history == False, then there is no need to save ------------
    if b_save_actions_history:
        date_time = datetime.utcnow()
        str_date_time = date_time.strftime("%Y%m%d_%H%M%S")
        action_description = f"graph_id:{input_graph.graph_id}," \
                             f"remove_node," \
                             f"node_index:{node_index}," \
                             f"{str_date_time}" \
                             f"\n"
        append_action_dataset_history(dataset_name, user_token, action_description)

    return output_graph


def add_edge(input_graph: torch_geometric.data.data.Data,
             new_edge_index_left: int,
             new_edge_index_right: int,
             new_edge_attr: Optional[np.array],
             dataset_name: str = None,
             user_token: str = None,
             b_save_actions_history: bool = False
             ) -> \
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
    :param dataset_name: Dataset name that specifies the name of the model too
    :param user_token: The user token that defines the sub-folder where the actions will be stored to
    :param b_save_actions_history: If True save the actions history in the dedicated file, else don't save the action.
    For b_save_actions_history==False is used for the graph reconstruction from the file.

    :return: The updated graph
    """

    ####################################################################################################################
    # [0.] Constraints/Requirements ====================================================================================
    ####################################################################################################################
    # [0.1.] Check the types of the input graph's object ---------------------------------------------------------------
    check_data_format_consistency(input_graph)

    # [0.2.] Check that nodes exist in the graph and check that the index of the left and right node is valid ----------
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

            if (new_edge_index_left, new_edge_index_right) in graph_edge_pairs or \
               (new_edge_index_right, new_edge_index_left) in graph_edge_pairs:

                raise ValueError(f"There is already an edge connecting node {new_edge_index_left} and "
                                 f"{new_edge_index_right}.\nMulti-graphs are not allowed.")

        # Add edge in the graph edge index ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
        input_graph_edge_index_left.append(new_edge_index_left)
        input_graph_edge_index_right.append(new_edge_index_right)

        # output_graph_edge_ids = np.append(output_graph_edge_ids, f"{new_edge_index_left}_{new_edge_index_right}")
        output_graph_edge_ids.append(f"{new_edge_index_left}_{new_edge_index_right}")

        output_graph_edge_index = torch.from_numpy(np.row_stack((input_graph_edge_index_left,
                                                                 input_graph_edge_index_right)))
    else:
        output_graph_edge_index = torch.from_numpy(np.array([[new_edge_index_left],
                                                             [new_edge_index_right]]))

    # [2.] Add the edge's features -------------------------------------------------------------------------------------
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

    # [4.] Take over the attribute labels of the edges, if they exist --------------------------------------------------
    if hasattr(input_graph, 'edge_attr_labels'):
        output_graph_edge_attr_labels = input_graph.edge_attr_labels
    else:
        output_graph_edge_attr_labels = None

    # [5.] Output graph ------------------------------------------------------------------------------------------------
    output_graph = Data(x=input_graph.x,
                        edge_index=output_graph_edge_index,
                        edge_attr=output_graph_edge_attr,
                        y=input_graph.y,
                        node_labels=input_graph.node_labels,
                        node_ids=input_graph.node_ids,
                        node_feature_labels=input_graph.node_feature_labels,
                        edge_ids=output_graph_edge_ids,
                        edge_attr_labels=output_graph_edge_attr_labels,
                        pos=output_pos,
                        graph_id=input_graph.graph_id
                        )
    check_data_format_consistency(output_graph)

    # [6.] Save/append the action in the local file if "b_save_actions_history" ----------------------------------------
    #       If in graph reconstruction phase b_save_actions_history == False, then there is no need to save ------------
    if b_save_actions_history:
        date_time = datetime.utcnow()
        str_date_time = date_time.strftime("%Y%m%d_%H%M%S")

        new_edge_attr_str = ","
        if new_edge_attr is not None:
            new_edge_attr_str = f"new_edge_attr: {np.array2string(new_edge_attr, separator=' ')},"

        action_description = f"graph_id: {input_graph.graph_id}," \
                             f"add_edge," \
                             f"new_edge_index_left:{new_edge_index_left}," \
                             f"new_edge_index_right:{new_edge_index_right}," \
                             + new_edge_attr_str + \
                             f"{str_date_time}" \
                             f"\n"
        append_action_dataset_history(dataset_name, user_token, action_description)

    return output_graph


def remove_edge(input_graph: torch_geometric.data.data.Data,
                edge_index_left: int,
                edge_index_right: int,
                dataset_name: str = None,
                user_token: str = None,
                b_save_actions_history: bool = False
                ) -> \
        torch_geometric.data.data.Data:
    """
    Remove an edge between two nodes. Check if the edge exists, remove the corresponding pair from "edge_index" and the
    attributes of this pair from the "edge_attr". All other fields stay unchanged.

    :param input_graph: Input graph
    :param edge_index_left: Index of left node of the edge
    :param edge_index_right: Index of right node of the edge
    :param dataset_name: Dataset name that specifies the name of the model too
    :param user_token: The user token that defines the sub-folder where the actions will be stored to
    :param b_save_actions_history: If True save the actions history in the dedicated file, else don't save the action.
    For b_save_actions_history==False is used for the graph reconstruction from the file.

    :return: The updated graph
    """

    ####################################################################################################################
    # [0.] Constraints/Requirements ====================================================================================
    ####################################################################################################################
    # [0.1.] Check the types of the input graph's object ---------------------------------------------------------------
    check_data_format_consistency(input_graph)

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

        if not (edge_index_left, edge_index_right) in graph_edge_pairs and \
                not (edge_index_right, edge_index_left) in graph_edge_pairs:
            raise ValueError(f"There is no edge connecting node {edge_index_left} and {edge_index_right}.\n")

        # # [2.] Remove the edge and its attributes --------------------------------------------------------------------
        if (edge_index_left, edge_index_right) in graph_edge_pairs:
            index_of_pair_to_delete = graph_edge_pairs.index((edge_index_left, edge_index_right))
        else:
            index_of_pair_to_delete = graph_edge_pairs.index((edge_index_right, edge_index_left))

        del input_graph_edge_index_left[index_of_pair_to_delete]
        del input_graph_edge_index_right[index_of_pair_to_delete]

        output_graph_edge_index = torch.from_numpy(np.row_stack((input_graph_edge_index_left,
                                                                 input_graph_edge_index_right))).to(dtype=torch.int64)
        # output_graph_edge_ids = np.delete(output_graph_edge_ids, index_of_pair_to_delete) >>>>>>>>>>>>>>>>>>>>>>>>>>>>
        del output_graph_edge_ids[index_of_pair_to_delete]

        if input_graph.edge_attr is not None:
            input_graph_edge_attr = input_graph.edge_attr.cpu().detach().numpy()
            input_graph_edge_attr = np.delete(input_graph_edge_attr, index_of_pair_to_delete, 0)
            output_graph_edge_attr = torch.from_numpy(input_graph_edge_attr)
        else:
            output_graph_edge_attr = None
    else:
        raise ValueError(f"There are no edges in the graph. This edge removal is per definition invalid.\n")

    # [3.] In the field position "pos" the position of the deleted node needs to be removed. ---------------------------
    output_pos = input_graph.pos

    # [4.] Take over the attribute labels of the edges, if they exist --------------------------------------------------
    if hasattr(input_graph, 'edge_attr_labels'):
        output_graph_edge_attr_labels = input_graph.edge_attr_labels
    else:
        output_graph_edge_attr_labels = None

    # [5.] Output graph ------------------------------------------------------------------------------------------------
    output_graph = Data(x=input_graph.x,
                        edge_index=output_graph_edge_index,
                        edge_attr=output_graph_edge_attr,
                        y=input_graph.y,
                        node_labels=input_graph.node_labels,
                        node_ids=input_graph.node_ids,
                        node_feature_labels=input_graph.node_feature_labels,
                        edge_ids=output_graph_edge_ids,
                        edge_attr_labels=output_graph_edge_attr_labels,
                        pos=output_pos,
                        graph_id=input_graph.graph_id
                        )

    check_data_format_consistency(output_graph)

    # [6.] Save/append the action in the local file if "b_save_actions_history" ----------------------------------------
    #       If in graph reconstruction phase b_save_actions_history == False, then there is no need to save ------------
    if b_save_actions_history:
        date_time = datetime.utcnow()
        str_date_time = date_time.strftime("%Y%m%d_%H%M%S")
        action_description = f"graph_id: {input_graph.graph_id}," \
                             f"remove_edge," \
                             f"edge_index_left:{edge_index_left}," \
                             f"edge_index_right:{edge_index_right}," \
                             f"{str_date_time}" \
                             f"\n"
        append_action_dataset_history(dataset_name, user_token, action_description)

    return output_graph


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
def add_feature_all_nodes(input_graph: torch_geometric.data.data.Data,
                          new_input_node_feature: np.array,
                          label: str,
                          dataset_name: str = None,
                          user_token: str = None,
                          b_save_actions_history: bool = False
                          ) -> torch_geometric.data.data.Data:
    """
    Add a feature in all nodes. Basically, that means that the features field of all the nodes "x" will have
    another column. The number of rows of the input feature should be equal to the number of nodes. If the node's
    features field "x" is empty then a one column array is created. The other fields stay unchanged.

    :param input_graph: Input graph
    :param new_input_node_feature: New input feature for nodes
    :param label: Feature label, which will be added
    :param dataset_name: Dataset name that specifies the name of the model too
    :param user_token: The user token that defines the sub-folder where the actions will be stored to
    :param b_save_actions_history: If True save the actions history in the dedicated file, else don't save the action.
    For b_save_actions_history==False is used for the graph reconstruction from the file.

    :return: The updated graph
    """

    ####################################################################################################################
    # [0.] Constraints/Requirements ====================================================================================
    ####################################################################################################################
    # [0.1.] Check the types of the input graph's object ---------------------------------------------------------------
    check_data_format_consistency(input_graph)

    # [0.2] Number of rows of the input feature should be equal to the number of nodes (each node must have a feature) -
    nodes_nr = input_graph.num_nodes
    assert nodes_nr == new_input_node_feature.shape[0], f"The number of nodes: {nodes_nr} in the graph is not equal " \
                                                        f"to the number of rows in the input feature: " \
                                                        f"{new_input_node_feature.shape[0]}. All nodes should have " \
                                                        f"the feature, since heterogeneous graphs are not allowed."

    # [0.3.] Check the type of node features ---------------------------------------------------------------------------
    assert new_input_node_feature.dtype == np.float32, f"The type of the node features must be: \"np.float32\".\n" \
                                                       f"Instead it is: {new_input_node_feature.dtype}"

    ####################################################################################################################
    # [1.] Add the feature to each node ================================================================================
    ####################################################################################################################
    input_graph_x = input_graph.x.cpu().detach().numpy()
    if input_graph_x is None:
        output_graph_x = np.array(new_input_node_feature)
    else:
        output_graph_x = np.column_stack((input_graph_x, new_input_node_feature))

    ####################################################################################################################
    # [2.] Output graph ================================================================================================
    ####################################################################################################################
    output_graph_node_feature_labels = copy.deepcopy(input_graph.node_feature_labels)
    output_graph_node_feature_labels.append(label)

    output_graph = Data(x=torch.from_numpy(output_graph_x),
                        edge_index=input_graph.edge_index,
                        edge_attr=input_graph.edge_attr,
                        y=input_graph.y,
                        node_labels=input_graph.node_labels,
                        node_ids=input_graph.node_ids,
                        node_feature_labels=output_graph_node_feature_labels,
                        edge_ids=input_graph.edge_ids,
                        edge_attr_labels=input_graph.edge_attr_labels,
                        pos=input_graph.pos,
                        graph_id=input_graph.graph_id
                        )

    check_data_format_consistency(output_graph)

    # [3.] Save/append the action in the local file if "b_save_actions_history" ----------------------------------------
    #       If in graph reconstruction phase b_save_actions_history == False, then there is no need to save ------------
    if b_save_actions_history:
        date_time = datetime.utcnow()
        str_date_time = date_time.strftime("%Y%m%d_%H%M%S")
        action_description = f"graph_id:{input_graph.graph_id}," \
                             f"add_feature_all_nodes," \
                             f"new_input_node_feature:{np.array2string(new_input_node_feature, separator=' ')}," \
                             f"label: {label}," \
                             f"{str_date_time}" \
                             f"\n"
        append_action_dataset_history(dataset_name, user_token, action_description)

    return output_graph


def remove_feature_all_nodes(input_graph: torch_geometric.data.data.Data,
                             removed_node_feature_idx: int,
                             dataset_name: str = None,
                             user_token: str = None,
                             b_save_actions_history: bool = False
                             ) -> \
        torch_geometric.data.data.Data:
    """
    Remove a feature in all the nodes. The features' field of all the nodes "x" will have one column less.
    It is presupposed that the node index is valid. If the node's features field "x" is empty then nothing is done.
    The other fields stay unchanged.

    :param input_graph: Input graph
    :param removed_node_feature_idx: Index of the column of the removed feature
    :param dataset_name: Dataset name that specifies the name of the model too
    :param user_token: The user token that defines the sub-folder where the actions will be stored to
    :param b_save_actions_history: If True save the actions history in the dedicated file, else don't save the action.
    For b_save_actions_history==False is used for the graph reconstruction from the file.

    :return: The updated graph
    """

    ####################################################################################################################
    # [0.] Constraints/Requirements ====================================================================================
    ####################################################################################################################

    # [0.1.] Check the types of the input graph's object ---------------------------------------------------------------
    check_data_format_consistency(input_graph)

    # [0.2] Check that the index of the deleted feature is valid -------------------------------------------------------

    assert input_graph.x is not None, "No nodes saved in the graphs, the \"x\" field is None"
    input_graph_x = input_graph.x.cpu().detach().numpy()
    node_features_nr = input_graph_x.shape[1]
    assert (0 <= removed_node_feature_idx) and (removed_node_feature_idx < node_features_nr), \
           f"The feature index to remove {removed_node_feature_idx} is not in accordance with the " \
           f"number of features {node_features_nr}"

    # [0.3.] Constraint: all features must have a label  ---------------------------------------------------------------
    assert len(input_graph.node_feature_labels) == node_features_nr, \
        f"The feature labels {len(input_graph.node_feature_labels)} must be equal " \
        f"to the number of features in the input graph"

    ####################################################################################################################
    # [1.] Remove the feature for each node  ---------------------------------------------------------------------------
    ####################################################################################################################

    output_graph_x = np.delete(input_graph_x, removed_node_feature_idx, 1)
    del input_graph.node_feature_labels[removed_node_feature_idx] # remove the label by index

    ####################################################################################################################
    # [2.] Output graph ------------------------------------------------------------------------------------------------
    ####################################################################################################################
    output_graph = Data(x=torch.from_numpy(output_graph_x),
                        edge_index=input_graph.edge_index,
                        edge_attr=input_graph.edge_attr,
                        y=input_graph.y,
                        node_labels=input_graph.node_labels,
                        node_ids=input_graph.node_ids,
                        node_feature_labels=input_graph.node_feature_labels,
                        edge_ids=input_graph.edge_ids,
                        edge_attr_labels=input_graph.edge_attr_labels,
                        pos=input_graph.pos,
                        graph_id=input_graph.graph_id
                        )
    check_data_format_consistency(output_graph)

    # [3.] Save/append the action in the local file if "b_save_actions_history" ----------------------------------------
    #       If in graph reconstruction phase b_save_actions_history == False, then there is no need to save ------------
    if b_save_actions_history:
        date_time = datetime.utcnow()
        str_date_time = date_time.strftime("%Y%m%d_%H%M%S")
        action_description = f"graph_id:{input_graph.graph_id}," \
                             f"remove_feature_all_nodes," \
                             f"removed_node_feature_idx:{removed_node_feature_idx}," \
                             f"{str_date_time}" \
                             f"\n"
        append_action_dataset_history(dataset_name, user_token, action_description)

    return output_graph


def add_feature_all_edges(input_graph: torch_geometric.data.data.Data,
                          new_input_edge_feature: np.array,
                          dataset_name: str = None,
                          user_token: str = None,
                          b_save_actions_history: bool = False
                          ) -> \
        torch_geometric.data.data.Data:
    """
    Add a feature in all of the edges. Basically, that means that the features field of all the edges "edge_attr"
    will have another column. The number of rows of the input attribute should be equal to the number of edges.
    If the edges's attributes field "edge_attr" is empty or None then a one column array is created.
    The other fields stay unchanged.

    :param input_graph: Input graph
    :param new_input_edge_feature: New input feature for the edges
    :param dataset_name: Dataset name that specifies the name of the model too
    :param user_token: The user token that defines the sub-folder where the actions will be stored to
    :param b_save_actions_history: If True save the actions history in the dedicated file, else don't save the action.
    For b_save_actions_history==False is used for the graph reconstruction from the file.
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
                        graph_id=input_graph.graph_id
                        )
    check_data_format_consistency(output_graph)

    # [4.] Save/append the action in the local file if "b_save_actions_history" ----------------------------------------
    #       If in graph reconstruction phase b_save_actions_history == False, then there is no need to save ------------
    if b_save_actions_history:
        date_time = datetime.utcnow()
        str_date_time = date_time.strftime("%Y%m%d_%H%M%S")
        action_description = f"graph_id:{input_graph.graph_id}," \
                             f"add_feature_all_edges," \
                             f"new_input_edge_feature:{np.array2string(new_input_edge_feature, separator=' ')}," \
                             f"{str_date_time}" \
                             f"\n"
        append_action_dataset_history(dataset_name, user_token, action_description)

    return output_graph


def remove_feature_all_edges(input_graph: torch_geometric.data.data.Data,
                             removed_edge_attribute_idx: int,
                             dataset_name: str = None,
                             user_token: str = None,
                             b_save_actions_history: bool = False
                             ) -> \
        torch_geometric.data.data.Data:
    """
    Remove a feature in all of the edges. The attributes field of all the edges "edge_attr" will have one column less.
    It is presupposed that the edge index is valid. If the edge's features field "edge_attr" is empty then nothing
    is done. The other fields stay unchanged.

    :param input_graph: Input graph
    :param removed_edge_attribute_idx: Index of the column of the removed attribute
    :param dataset_name: Dataset name that specifies the name of the model too
    :param user_token: The user token that defines the sub-folder where the actions will be stored to
    :param b_save_actions_history: If True save the actions history in the dedicated file, else don't save the action.
    For b_save_actions_history==False is used for the graph reconstruction from the file.

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
                        graph_id=input_graph.graph_id
                        )
    check_data_format_consistency(output_graph)

    # [4.] Save/append the action in the local file if "b_save_actions_history" ----------------------------------------
    #       If in graph reconstruction phase b_save_actions_history == False, then there is no need to save ------------
    if b_save_actions_history:
        date_time = datetime.utcnow()
        str_date_time = date_time.strftime("%Y%m%d_%H%M%S")
        action_description = f"graph_id: {input_graph.graph_id}," \
                             f"remove_feature_all_edges," \
                             f"removed_edge_attribute_idx:{removed_edge_attribute_idx}," \
                             f"{str_date_time}" \
                             f"\n"
        append_action_dataset_history(dataset_name, user_token, action_description)

    return output_graph
