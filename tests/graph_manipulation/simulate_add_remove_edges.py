"""
    Simulate the addition and removal of edges

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-11-10
"""

import copy
import random
import pytest

import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.utils.convert import to_networkx

from actionable.graph_actions import add_edge, remove_edge
from testing_utils.testing_data_generation import generate_data_set


def add_edges_simulation(input_graph: torch_geometric.data.data.Data):
    """
    Simulate the addition of valid edges in an existing graph.
    Properties of the graph after any edge addition:

    [1.] The number of edges is increased
    [2.] The number of nodes, number of node features, number of edge features must stay the same
    """

    # [0.] Perform edge additions --------------------------------------------------------------------------------------
    #      Only add valid edges to the graph (non existent with valid node indices) ------------------------------------
    edge_addition_actions_max_nr = 100
    edge_addition_actions_nr = random.randint(1, edge_addition_actions_max_nr)
    max_number_nodes = input_graph.node_stores[0].num_nodes
    added_edges = 0
    graph_edge_pairs = 0

    updated_graph = copy.deepcopy(input_graph)
    edge_features_nr = input_graph.edge_attr

    for edge_addition_action in range(edge_addition_actions_nr):
        if edge_features_nr is not None:
            edge_features = np.random.randn(1, edge_features_nr.shape[1])
        else:
            edge_features = None

        edge_index_left = np.random.randint(max_number_nodes)
        edge_index_right = np.random.randint(max_number_nodes)

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

    # [1.] What should change ------------------------------------------------------------------------------------------
    # 1.1. Number of edges must equal actions --------------------------------------------------------------------------
    assert updated_graph.edge_stores[0].num_edges == input_graph.edge_stores[0].num_edges + added_edges, \
        f"The number of edges in the updated graph {updated_graph.edge_stores[0].num_edges}" \
        f" must equal to the number of edges " \
        f"in the input graph {input_graph.edge_stores[0].num_edges} and added edges {added_edges}."

    # 1.2. Number of edges -------------------------------------------------------------------------------------
    if input_graph.edge_attr is not None:
        assert input_graph.edge_attr.shape[0] < updated_graph.edge_attr.shape[0], \
            f"The number of edges of the input graph must be less than the one of the updated graph."

    # [2.] What should stay constant -----------------------------------------------------------------------------------
    # 2.1. Number of nodes ---------------------------------------------------------------------------------------------
    assert updated_graph.x.shape[0] == input_graph.x.shape[0], \
        f"The nodes of the input graph must be the same as the ones of the updated graph."

    # 2.2. Number of node features -------------------------------------------------------------------------------------
    assert input_graph.x.shape[1] == updated_graph.x.shape[1], \
        f"The number of the node features of the input graph {input_graph.x.shape[1]} must equal " \
        f"to the number of node features of the updated graph {updated_graph.x.shape[1]}."

    # 2.3. Edge features -----------------------------------------------------------------------------------------------
    if input_graph.edge_attr is not None:
        assert input_graph.edge_attr.shape[1] == updated_graph.edge_attr.shape[1], \
            f"The edges of the input graph must be the same as the one of the updated graph."

    # 2.4. Classes / Labels --------------------------------------------------------------------------------------------
    if input_graph.y is not None:
        assert torch.equal(input_graph.y, updated_graph.y), \
            f"The classes/labels of the input graph must be the same as the one of the updated graph."


def add_edges_simulation_invalid(input_graph: torch_geometric.data.data.Data):
    """
    Simulate the addition of invalid edges in an existing graph.
    Properties of the graph after any edge addition:

    [1.] The number of edges is increased
    [2.] The number of nodes, number of node features, number of edge features must stay the same
    """

    # [0.] Perform edge additions --------------------------------------------------------------------------------------
    #      Test different combinations of valid additions and invalid ones (node index, feature dimension) -------------
    edge_addition_actions_max_nr = 100
    edge_addition_actions_nr = random.randint(1, edge_addition_actions_max_nr)
    max_number_nodes = input_graph.node_stores[0].num_nodes
    added_edges = 0
    failed_attempts = 0

    edge_features_nr = input_graph.edge_attr
    updated_graph = copy.deepcopy(input_graph)

    for edge_addition_action in range(edge_addition_actions_nr):
        if edge_features_nr is not None:
            edge_features = np.random.randn(1, edge_features_nr.shape[1])
        else:
            edge_features = None
        edge_index_left = np.random.randint(max_number_nodes * 2)
        edge_index_right = np.random.randint(max_number_nodes * 2)

        updated_graph_edge_index = updated_graph.edge_index.numpy()
        updated_graph_edge_index_left = list(updated_graph_edge_index[0, :])
        updated_graph_edge_index_right = list(updated_graph_edge_index[1, :])

        if edge_index_right > max_number_nodes or edge_index_left > max_number_nodes:
            try:
                updated_graph = add_edge(updated_graph, edge_index_left, edge_index_right, edge_features)
                if len(updated_graph_edge_index_left) > 0 and len(updated_graph_edge_index_right) > 0:
                    graph_edge_pairs = list(map(lambda x: (x[0], x[1]),
                                            list(zip(updated_graph_edge_index_left, updated_graph_edge_index_right))))
                    assert (edge_index_left, edge_index_right) not in graph_edge_pairs, \
                        f"Multi-graphs are not allowed!"
                added_edges += 1
            except AssertionError:
                failed_attempts += 1
        else:
            try:
                invalid_features = np.random.randn(1, 20)
                updated_graph = add_edge(updated_graph, edge_index_left, edge_index_right, invalid_features)
                if len(updated_graph_edge_index_left) > 0 and len(updated_graph_edge_index_right) > 0:
                    graph_edge_pairs = list(map(lambda x: (x[0], x[1]),
                                            list(zip(updated_graph_edge_index_left, updated_graph_edge_index_right))))
                    assert (edge_index_left, edge_index_right) not in graph_edge_pairs, \
                        f"Multi-graphs are not allowed!"
                added_edges += 1
            except AssertionError:
                failed_attempts += 1

    # [1.] What should change ------------------------------------------------------------------------------------------
    # 1.1. Number of edges must equal actions --------------------------------------------------------------------------
    assert updated_graph.edge_stores[0].num_edges == input_graph.edge_stores[0].num_edges + added_edges, \
        f"The number of edges in the updated graph {updated_graph.edge_stores[0].num_edges}" \
        f" must equal to the number of edges " \
        f"in the input graph {input_graph.edge_stores[0].num_edges} and added edges {added_edges}."

    assert edge_addition_actions_nr == added_edges + failed_attempts, \
        f"The number of actions {edge_addition_actions_nr} must equal to the number of valid attempts  " \
        f"{added_edges} and failed attempts {failed_attempts}."

    # 1.2. Number of edges -------------------------------------------------------------------------------------
    if input_graph.edge_attr is not None:
        assert input_graph.edge_attr.shape[0] < updated_graph.edge_attr.shape[0], \
            f"The number of edges of the input graph must be less than the one of the updated graph."

    # [2.] What should stay constant -----------------------------------------------------------------------------------
    # 2.1. Number of nodes ---------------------------------------------------------------------------------------------
    assert updated_graph.x.shape[0] == input_graph.x.shape[0], \
        f"The nodes of the input graph must be the same as the ones of the updated graph."

    # 2.2. Number of node features -------------------------------------------------------------------------------------
    assert input_graph.x.shape[1] == updated_graph.x.shape[1], \
        f"The number of the node features of the input graph {input_graph.x.shape[1]} must equal " \
        f"to the number of node features of the updated graph {updated_graph.x.shape[1]}."

    # 2.3. Edge features -----------------------------------------------------------------------------------------------
    if input_graph.edge_attr is not None:
        assert input_graph.edge_attr.shape[1] == updated_graph.edge_attr.shape[1], \
            f"The edges of the input graph must be the same as the one of the updated graph."

    # 2.4. Classes / Labels --------------------------------------------------------------------------------------------
    if input_graph.y is not None:
        assert torch.equal(input_graph.y, updated_graph.y), \
            f"The classes/labels of the input graph must be the same as the one of the updated graph."


def remove_edges_simulation(input_graph: torch_geometric.data.data.Data):
    """
    Simulate the removal of edges in an existing graph.
    Properties of the graph after any edge removal:

    [1.] The number of edges is decreased
    [2.] The number of nodes, number of node features, number of edge features must stay the same
    """

    # [0.] Perform edge removals ---------------------------------------------------------------------------------------
    #      Only remove valid edges (existent, valid indices) -----------------------------------------------------------
    edge_removal_actions_max_nr = 100
    edge_removal_actions_nr = random.randint(1, edge_removal_actions_max_nr)
    max_number_nodes = input_graph.node_stores[0].num_nodes
    removed_edges = 0
    graph_edge_pairs = 0

    updated_graph = copy.deepcopy(input_graph)

    for edge_removal_action in range(edge_removal_actions_nr):
        edge_index_left = np.random.randint(max_number_nodes)
        edge_index_right = np.random.randint(max_number_nodes)

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
    # 1.1. Number of edges must equal actions --------------------------------------------------------------------------
    assert updated_graph.edge_stores[0].num_edges == input_graph.edge_stores[0].num_edges - removed_edges, \
        f"The number of edges in the updated graph {updated_graph.edge_stores[0].num_edges}" \
        f" must equal to the number of edges " \
        f"in the input graph {input_graph.edge_stores[0].num_edges} and removed edges {removed_edges}."

    # [2.] What should stay constant -----------------------------------------------------------------------------------
    # 2.1. Number of nodes ---------------------------------------------------------------------------------------------
    assert updated_graph.x.shape[0] == input_graph.x.shape[0], \
        f"The nodes of the input graph must be the same as the ones of the updated graph."

    # 2.2. Number of node features -------------------------------------------------------------------------------------
    assert input_graph.x.shape[1] == updated_graph.x.shape[1], \
        f"The number of the node features of the input graph {input_graph.x.shape[1]} must equal " \
        f"to the number of node features of the updated graph {updated_graph.x.shape[1]}."

    # 2.3. Edge features -----------------------------------------------------------------------------------------------
    if input_graph.edge_attr is not None:
        assert input_graph.edge_attr.shape[1] == updated_graph.edge_attr.shape[1], \
            f"The edges of the input graph must be the same as the one of the updated graph."

    # 2.4. Classes / Labels --------------------------------------------------------------------------------------------
    if input_graph.y is not None:
        assert torch.equal(input_graph.y, updated_graph.y), \
            f"The classes/labels of the input graph must be the same as the one of the updated graph."


def remove_edges_simulation_invalid(input_graph: torch_geometric.data.data.Data):
    """
    Simulate the removal of invalid/ not existing edges in an existing graph.
    Properties of the graph after any edge removal:

    [1.] The number of edges is decreased
    [2.] The number of nodes, number of node features, number of edge features must stay the same
    """

    # [0.] Perform edge removals ---------------------------------------------------------------------------------------
    #      Test valid and invalid edge removals ------------------------------------------------------------------------
    edge_removal_actions_max_nr = 100
    edge_removal_actions_nr = random.randint(1, edge_removal_actions_max_nr)
    max_number_nodes = input_graph.node_stores[0].num_nodes
    removed_edges = 0
    failed_attempts = 0

    updated_graph = copy.deepcopy(input_graph)

    for edge_removal_action in range(edge_removal_actions_nr):
        edge_index_left = np.random.randint(max_number_nodes * 2)
        edge_index_right = np.random.randint(max_number_nodes * 2)

        try:
            updated_graph = remove_edge(updated_graph, edge_index_left, edge_index_right)
            removed_edges += 1
        except AssertionError:
            failed_attempts += 1

    # [1.] What should change ------------------------------------------------------------------------------------------
    # 1.1. Number of edges must equal actions --------------------------------------------------------------------------
    assert updated_graph.edge_stores[0].num_edges == input_graph.edge_stores[0].num_edges - removed_edges, \
        f"The number of edges in the updated graph {updated_graph.edge_stores[0].num_edges}" \
        f" must equal to the number of edges " \
        f"in the input graph {input_graph.edge_stores[0].num_edges} and removed edges {removed_edges}."

    assert edge_removal_actions_nr == removed_edges + failed_attempts, \
        f"The number of actions {edge_removal_actions_nr} must equal to the number of valid attempts  " \
        f"{removed_edges} and failed attempts {failed_attempts}."

    # [2.] What should stay constant -----------------------------------------------------------------------------------
    # 2.1. Number of nodes ---------------------------------------------------------------------------------------------
    assert updated_graph.x.shape[0] == input_graph.x.shape[0], \
        f"The nodes of the input graph must be the same as the ones of the updated graph."

    # 2.2. Number of node features -------------------------------------------------------------------------------------
    assert input_graph.x.shape[1] == updated_graph.x.shape[1], \
        f"The number of the node features of the input graph {input_graph.x.shape[1]} must equal " \
        f"to the number of node features of the updated graph {updated_graph.x.shape[1]}."

    # 2.3. Edge features -----------------------------------------------------------------------------------------------
    if input_graph.edge_attr is not None:
        assert input_graph.edge_attr.shape[1] == updated_graph.edge_attr.shape[1], \
            f"The edges of the input graph must be the same as the one of the updated graph."

    # 2.4. Classes / Labels --------------------------------------------------------------------------------------------
    if input_graph.y is not None:
        assert torch.equal(input_graph.y, updated_graph.y), \
            f"The classes/labels of the input graph must be the same as the one of the updated graph."


def remove_edges_simulation_all(input_graph: torch_geometric.data.data.Data):
    """
    Simulate the removal of all edges in an existing graph.
    Properties of the graph after any edge removal:

    [1.] No edges are in the graph
    [2.] The number of nodes, number of node features, number of edge features must stay the same
    """

    # [0.] Perform edge removals ---------------------------------------------------------------------------------------
    input_graph_edge_index = input_graph.edge_index.numpy()
    input_graph_edge_index_left = list(input_graph_edge_index[0, :])
    input_graph_edge_index_right = list(input_graph_edge_index[1, :])

    if len(input_graph_edge_index_left) > 0 and len(input_graph_edge_index_right) > 0:
        graph_edge_pairs = list(map(lambda x: (x[0], x[1]),
                                    list(zip(input_graph_edge_index_left, input_graph_edge_index_right))))

    updated_graph = copy.deepcopy(input_graph)

    for (edge_index_left, edge_index_right) in graph_edge_pairs:
        updated_graph = remove_edge(updated_graph, edge_index_left, edge_index_right)

    # [1.] What should change ------------------------------------------------------------------------------------------
    assert updated_graph.edge_stores[0].num_edges == 0, \
        f"The number of edges in the updated graph {updated_graph.edge_stores[0].num_edges}" \
        f" must equal to 0."

    # [2.] What should stay constant -----------------------------------------------------------------------------------
    # 2.1. Number of nodes ---------------------------------------------------------------------------------------------
    assert updated_graph.x.shape[0] == input_graph.x.shape[0], \
        f"The nodes of the input graph must be the same as the ones of the updated graph."

    # 2.2. Number of node features -------------------------------------------------------------------------------------
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

    # [3.] Check that graph is unconnected -----------------------------------------------------------------------------
    updated_graph_nx = to_networkx(updated_graph, to_undirected=True)
    assert not nx.is_connected(updated_graph_nx), \
        f"Since there are no edges drawn, the graph must be unconnected."


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

    # [3.] Make some edge additions ------------------------------------------------------------------------------------
    add_edges_simulation(selected_graph)
    add_edges_simulation_invalid(selected_graph)

    # [4.] Remove edges ------------------------------------------------------------------------------------------------
    remove_edges_simulation(selected_graph)
    remove_edges_simulation_invalid(selected_graph)
    remove_edges_simulation_all(selected_graph)


def test_barabasi_simulation():
    simulate_actions(dataset_names[0])


def test_kirc_simulation():
    simulate_actions(dataset_names[1])

