"""
    Simulate the addition and removal of nodes and edges

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-11-10
"""

import copy
import random
import pytest

import numpy as np
import torch
import torch_geometric
from torch_geometric.transforms import NormalizeFeatures

from actionable.graph_actions import add_edge, remove_edge
from actionable.graph_actions import add_node, remove_node
from testing_utils.testing_data_generation import generate_data_set


def add_remove_nodes_edges_simulation_valid(input_graph: torch_geometric.data.data.Data):
    """
    Add and remove nodes and edges intertwined

    - Check in the list of edges that you don't add an edge that already exists.
    - Check in the list of edges that you don't remove an edge does not exist.
    - Don't remove the last node in the graph.
    """

    actions_max_nr = 100
    # [1.] Random number of node and edge additions --------------------------------------------------------------------
    addition_actions_nr = random.randint(1, actions_max_nr)
    node_features_nr = input_graph.x.shape[1]
    edge_features_nr = input_graph.edge_attr
    added_edges = 0
    graph_edge_pairs = 0

    failed_node_removals = 0

    updated_graph = copy.deepcopy(input_graph)
    for action in range(addition_actions_nr):
        max_number_nodes = updated_graph.node_stores[0].num_nodes
        edge_index_left = np.random.randint(max_number_nodes)
        edge_index_right = np.random.randint(max_number_nodes)
        node_features = np.random.randn(1, node_features_nr)
        if edge_features_nr is not None:
            edge_features = np.random.randn(1, edge_features_nr.shape[1])
        else:
            edge_features = None

        # 1.1 node additions -------------------------------------------------------------------------------------------
        updated_graph = add_node(updated_graph, node_features, "", "node_" + str(action))

        # 1.2 edge additions -------------------------------------------------------------------------------------------
        updated_graph_edge_index = updated_graph.edge_index.numpy()
        updated_graph_edge_index_left = list(updated_graph_edge_index[0, :])
        updated_graph_edge_index_right = list(updated_graph_edge_index[1, :])
        if len(updated_graph_edge_index_left) > 0 and len(updated_graph_edge_index_right) > 0:
            graph_edge_pairs = list(map(lambda x: (x[0], x[1]),
                                        list(zip(updated_graph_edge_index_left, updated_graph_edge_index_right))))

        if edge_index_left != edge_index_right and (edge_index_left, edge_index_right) not in graph_edge_pairs\
                and (edge_index_right, edge_index_left) not in graph_edge_pairs:
            updated_graph = add_edge(updated_graph, edge_index_left, edge_index_right, edge_features)
            added_edges += 1

    # [2.] Random number of node and edge removal ----------------------------------------------------------------------
    #      The number of edges in the end can be zero (0), but the minimum number of nodes must be one (1) -------------
    removal_actions_nr = random.randint(1, actions_max_nr)
    removed_edges = 0

    for action in range(removal_actions_nr):
        max_number_nodes = updated_graph.x.shape[0]
        edge_index_left = np.random.randint(max_number_nodes)
        edge_index_right = np.random.randint(max_number_nodes)
        node_index = np.random.randint(0, max_number_nodes)

        # 2.1 node removals --------------------------------------------------------------------------------------------
        try:
            label = updated_graph.node_labels[node_index]
            updated_graph = remove_node(updated_graph, node_index, label)
        except (KeyError, IndexError) as e:
            failed_node_removals += 1

        # 2.2 edge removals --------------------------------------------------------------------------------------------
        updated_graph_edge_index = updated_graph.edge_index.numpy()
        updated_graph_edge_index_left = list(updated_graph_edge_index[0, :])
        updated_graph_edge_index_right = list(updated_graph_edge_index[1, :])
        if len(updated_graph_edge_index_left) > 0 and len(updated_graph_edge_index_right) > 0:
            graph_edge_pairs = list(map(lambda x: (x[0], x[1]),
                                        list(zip(updated_graph_edge_index_left, updated_graph_edge_index_right))))

        if edge_index_left != edge_index_right and (edge_index_left, edge_index_right) in graph_edge_pairs:
            updated_graph = remove_edge(updated_graph, edge_index_left, edge_index_right)
            removed_edges += 1

    # [1.] What should change ------------------------------------------------------------------------------------------
    assert updated_graph.x.shape[0] == input_graph.x.shape[0] + addition_actions_nr - removal_actions_nr + failed_node_removals, \
        f"The number of nodes in the updated graph {updated_graph.x.shape[0]} must equal to the number of nodes" \
        f"in the input graph {input_graph.x.shape[0]} and added nodes {addition_actions_nr} minus removed nodes " \
        f"{removal_actions_nr} (+ number of failed node removals {failed_node_removals})."

    assert updated_graph.x.shape[0] >= 1, \
        f"The minimal number of nodes in the updated graph {updated_graph.x.shape[0]} must be 1."

    assert updated_graph.edge_stores[0].num_edges <= \
        input_graph.edge_stores[0].num_edges + added_edges - removed_edges, \
        f"The number of edges in the updated graph {updated_graph.edge_stores[0].num_edges}" \
        f" must less or equal to the number of edges in the input graph {input_graph.edge_stores[0].num_edges} " \
        f"and added edges {added_edges} minus removed edges {removed_edges}."

    assert updated_graph.edge_stores[0].num_edges >= 0, \
        f"The minimum number of edges in the updated graph {updated_graph.edge_stores[0].num_edges} must be 0."

    # [2.] What should stay constant -----------------------------------------------------------------------------------
    # 2.1. Number of node features -------------------------------------------------------------------------------------
    assert input_graph.x.shape[1] == updated_graph.x.shape[1], \
        f"The number of the node features of the input graph {input_graph.x.shape[1]} must equal " \
        f"to the number of node features of the updated graph {updated_graph.x.shape[1]}."

    # 2.3. Edge features -----------------------------------------------------------------------------------------------
    if input_graph.edge_attr is not None:
        assert input_graph.edge_attr.shape[1] == updated_graph.edge_attr.shape[1], \
            f"The edges of the input graph must be the same as the one of the updated graph."

    # 2.4. Classes / Labels --------------------------------------------------------------------------------------------
    assert torch.equal(input_graph.y, updated_graph.y), \
        f"The classes/labels of the input graph must be the same as the one of the updated graph."


########################################################################################################################
# MAIN =================================================================================================================
########################################################################################################################

# [1.] Graphs dataset that was used in the GNN task --------------------------------------------------------------------
dataset_names = ["Barabasi-Albert Dataset", "Kirc Dataset"]


def simulate_actions(dataset_name):
    dataset = generate_data_set(dataset_name)

    # [2.] Perform the first task (atm: node classification) with the GNN ----------------------------------------------
    graph_idx = 0
    patient_graph = dataset[str(graph_idx)]
    selected_graph = patient_graph[str(graph_idx)]  # Get the selected graph object. -----------------------------------

    # Add y to Barabasi dataset ----------------------------------------------------------------------------------------
    if dataset_name == dataset_names[0]:
        selected_graph.y = torch.from_numpy(np.array([1, 2, 3]))

    # [3.] Make changes ------------------------------------------------------------------------------------------------
    add_remove_nodes_edges_simulation_valid(selected_graph)


def test_barabasi_simulation():
    simulate_actions(dataset_names[0])


def test_kirc_simulation():
    simulate_actions(dataset_names[1])
