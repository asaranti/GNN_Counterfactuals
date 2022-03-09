"""
    Simulate the addition and removal of nodes

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

from actionable.graph_actions import add_node, remove_node
from testing_utils.testing_data_generation import generate_data_set


def add_nodes_simulation(input_graph: torch_geometric.data.data.Data):
    """
    Simulate the addition of nodes in an existing graph.
    Properties of the graph after any node addition:

    [1.] The number of nodes is increased
    [2.] Number of edges, number of node features, number of edge features, classes must stay the same
    [3.] Since no edges are drawn, the graph must be unconnected

    :param input_graph: Input graph that will be manipulated
    """

    # [0.] Perform node additions --------------------------------------------------------------------------------------
    node_addition_actions_max_nr = 100
    node_addition_actions_nr = random.randint(1, node_addition_actions_max_nr)

    node_features_nr = input_graph.x.shape[1]

    updated_graph = copy.deepcopy(input_graph)
    for node_addition_action in range(node_addition_actions_nr):

        node_features = np.random.randn(1, node_features_nr)
        updated_graph = add_node(updated_graph, node_features, "", node_addition_action)

    # [1.] What should change ------------------------------------------------------------------------------------------
    assert updated_graph.x.shape[0] == input_graph.x.shape[0] + node_addition_actions_nr, \
        f"The number of nodes in the updated graph {updated_graph.x.shape[0]} must equal to the number of nodes" \
        f"in the input graph {input_graph.x.shape[0]} and added nodes {node_addition_actions_nr}."

    # [2.] What should stay constant -----------------------------------------------------------------------------------
    # 2.1. Number of edges ---------------------------------------------------------------------------------------------
    assert torch.equal(input_graph.edge_index, updated_graph.edge_index), \
        f"The edges of the input graph must be the same as the one of the updated graph."

    # 2.2. Number of node features -------------------------------------------------------------------------------------
    assert input_graph.x.shape[1] == updated_graph.x.shape[1], \
        f"The number of the node features of the input graph {input_graph.x.shape[1]} must equal " \
        f"to the number of node features of the updated graph {updated_graph.x.shape[1]}."

    # 2.3. Edge features -----------------------------------------------------------------------------------------------
    if input_graph.edge_attr is not None:
        assert torch.equal(input_graph.edge_attr, updated_graph.edge_attr), \
            f"The edges of the input graph must be the same as the one of the updated graph."

    # 2.4. Classes / Labels --------------------------------------------------------------------------------------------
    if input_graph.y is not None:
        assert torch.equal(input_graph.y, updated_graph.y), \
            f"The classes/labels of the input graph must be the same as the one of the updated graph."

    # [3.] Check that graph is unconnected -----------------------------------------------------------------------------
    updated_graph_nx = to_networkx(updated_graph, to_undirected=True)
    assert not nx.is_connected(updated_graph_nx), \
        f"Since there are no edges drawn, the graph must be unconnected."


def remove_nodes_simulation(input_graph: torch_geometric.data.data.Data):
    """
    Simulate the removal of nodes in an existing graph.
    Properties of the graph after any node removal:

    [1.] The number of nodes is decreased
    [2.] The number of edges of the new graph is <= number of edges of the old graph
    [3.] Number of node features, number of edge features must stay the same
    """

    # [0.] Perform node removals ---------------------------------------------------------------------------------------
    node_removal_actions_max_nr = 100
    node_removal_actions_nr = random.randint(1, node_removal_actions_max_nr)
    max_number_nodes = input_graph.node_stores[0].num_nodes
    removed_nodes = 0
    failed_removals = 0

    updated_graph = copy.deepcopy(input_graph)
    for node_removal_action in range(node_removal_actions_nr):
        node_index = np.random.randint(max_number_nodes * 2)

        try:
            label = updated_graph.node_labels[node_index]
            updated_graph = remove_node(updated_graph, node_index, label)
            removed_nodes += 1
        except (AssertionError, KeyError, IndexError) as e:
            failed_removals += 1

    # [1.] What should change ------------------------------------------------------------------------------------------
    assert updated_graph.x.shape[0] == input_graph.x.shape[0] - removed_nodes, \
        f"The number of nodes in the updated graph {updated_graph.x.shape[0]} must equal to the number of nodes" \
        f"in the input graph {input_graph.x.shape[0]} and removed nodes {removed_nodes}."

    assert node_removal_actions_nr == removed_nodes + failed_removals, \
        f"The number of actions {node_removal_actions_nr} must equal to the number of valid attempts  " \
        f"{removed_nodes} and failed attempts {failed_removals}."

    # [2.] What should stay constant -----------------------------------------------------------------------------------
    # 2.1. Number of node features -------------------------------------------------------------------------------------
    assert input_graph.x.shape[1] == updated_graph.x.shape[1], \
        f"The number of the node features of the input graph {input_graph.x.shape[1]} must equal " \
        f"to the number of node features of the updated graph {updated_graph.x.shape[1]}."

    # 2.2. Edge features -----------------------------------------------------------------------------------------------
    if input_graph.edge_attr is not None:
        assert torch.equal(input_graph.edge_attr, updated_graph.edge_attr), \
            f"The edges of the input graph must be the same as the one of the updated graph."

    # 2.3. Classes / Labels --------------------------------------------------------------------------------------------
    if input_graph.y is not None:
        assert torch.equal(input_graph.y, updated_graph.y), \
            f"The classes/labels of the input graph must be the same as the one of the updated graph."

    # 2.4 Number of edges (removed through call of UI) -----------------------------------------------------------------
    assert updated_graph.edge_stores[0].num_edges == input_graph.edge_stores[0].num_edges, \
        f"The number of edges in the updated graph {updated_graph.edge_stores[0].num_edges} " \
        f"must be equal to the number of edges in the input graph {input_graph.edge_stores[0].num_edges}."


def remove_nodes_simulation_all(input_graph: torch_geometric.data.data.Data):
    """
    Simulate the removal of all nodes in an existing graph.
    Properties of the graph after any node removal:

    [1.] The number of nodes is 0
    [2.] The number of edges of the new graph is 0
    [3.] Number of node features, number of edge features must stay the same
    """

    # [0.] Perform node removals ---------------------------------------------------------------------------------------
    updated_graph = copy.deepcopy(input_graph)

    for node in range(input_graph.num_nodes-1, -1, -1):
        label = updated_graph.node_labels[node]
        updated_graph = remove_node(updated_graph, node, label)

    # [1.] What should change ------------------------------------------------------------------------------------------
    assert updated_graph.x.shape[0] == 0, \
        f"The number of nodes in the updated graph {updated_graph.x.shape[0]} must equal to 0."

    # [2.] What should stay constant -----------------------------------------------------------------------------------
    # 2.1. Number of node features -------------------------------------------------------------------------------------
    assert input_graph.x.shape[1] == updated_graph.x.shape[1], \
        f"The number of the node features of the input graph {input_graph.x.shape[1]} must equal " \
        f"to the number of node features of the updated graph {updated_graph.x.shape[1]}."

    # 2.2. Edge features -----------------------------------------------------------------------------------------------
    if input_graph.edge_attr is not None:
        assert torch.equal(input_graph.edge_attr, updated_graph.edge_attr), \
            f"The edges of the input graph must be the same as the one of the updated graph."

    # 2.3. Classes / Labels --------------------------------------------------------------------------------------------
    assert torch.equal(input_graph.y, updated_graph.y), \
        f"The classes/labels of the input graph must be the same as the one of the updated graph."

    # 2.4 Number of edges (removed through call of UI) -----------------------------------------------------------------
    assert updated_graph.edge_stores[0].num_edges == input_graph.edge_stores[0].num_edges, \
        f"The number of edges in the updated graph {updated_graph.edge_stores[0].num_edges} must be equal to the " \
        f"number of edges in the input graph {input_graph.edge_stores[0].num_edges}."


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

    # [3.] Make some node additions ------------------------------------------------------------------------------------
    add_nodes_simulation(selected_graph)

    # [4.] Remove nodes ------------------------------------------------------------------------------------------------
    remove_nodes_simulation(selected_graph)
    remove_nodes_simulation_all(selected_graph)


def test_barabasi_simulation():
    simulate_actions(dataset_names[0])


def test_kirc_simulation():
    simulate_actions(dataset_names[1])
