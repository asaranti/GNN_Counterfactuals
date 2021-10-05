"""
    All possible graph actions

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-10-05
"""


import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data


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
    assert input_graph.x is None, "No nodes saved in the graphs, the \"x\" field is None"
    assert node_index < 0 or node_index >= input_graph.num_nodes, \
        f"The index of the node {node_index} is not in accordance with the number of nodes {input_graph.num_nodes}"

    # [1.] The corresponding line needs to be removed from the "x" field of the Data class, ----------------------------
    #      corresponding to node features. -----------------------------------------------------------------------------
    input_graph_x = input_graph.x.numpy()
    output_graph_x = np.delete(input_graph_x, node_index, 0)

    # [2.] All edges in which the node participates need to be deleted. ------------------------------------------------
    #      The corresponding pairs in the "edge_index" need to be removed. ---------------------------------------------
    input_graph_edge_index = input_graph.edge_index.numpy()
    input_graph_edge_index_left = input_graph_edge_index[0, :]
    input_graph_edge_index_right = input_graph_edge_index[1, :]
    # TODO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # [3.] The features of all the edges that are deleted in the previous step -----------------------------------------
    #          must also be deleted. -----------------------------------------------------------------------------------

    # [4.] The field "y" must not be deleted. --------------------------------------------------------------------------
    #      Nothing to do here

    # [5.] In the field position "pos" the position of the deleted node needs to be removed. ---------------------------
    # TODO: Implement it !!!!! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # [6.] Output graph ------------------------------------------------------------------------------------------------
    output_graph = Data(x=torch.from_numpy(output_graph_x),
                        edge_index=output_graph_edge_index,
                        edge_attr=torch.from_numpy(output_graph_egde_attr),
                        y=input_graph.y, dtype=torch.long)
    return output_graph

# def add_node(input_graph: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data:

# def add_edge(input_graph: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data:

# def remove_edge(input_graph: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data:

# def add_feature_all_nodes(input_graph: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data:

# def remove_feature_all_nodes(input_graph: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data:

# def add_feature_all_edges(input_graph: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data:

# def remove_feature_all_edges(input_graph: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data:
