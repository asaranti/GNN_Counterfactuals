"""
    Flask application instance

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-10-18
"""

from flask import Flask, request

import numpy as np
from torch_geometric.datasets import TUDataset

from actionable.graph_actions import add_node, add_edge, remove_node, remove_edge, \
    add_feature_all_nodes, remove_feature_all_nodes, add_feature_all_edges, remove_feature_all_edges
from app_utils.jsonification import graph_to_json

########################################################################################################################
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
########################################################################################################################
app = Flask(__name__)


# Start: Index ---------------------------------------------------------------------------------------------------------
@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"


dataset = TUDataset(root='data/TUDataset', name='MUTAG')
graph_idx = 0


########################################################################################################################
# [0.1.] Get the first graph (default) in JSON format ==================================================================
########################################################################################################################
@app.route('/first_graph', methods=['GET'])
def get_first_graph():
    """
    Get the first graph as default

    :return:
    """

    graph_json = graph_to_json(dataset[graph_idx])

    return graph_json


########################################################################################################################
# [0.2.] Get the number of graphs in the dataset =======================================================================
########################################################################################################################
@app.route('/graph_dataset_size', methods=['GET'])
def get_graph_dataset_size():
    """
    Get graphs dataset size

    :return:
    """

    graphs_nr = len(dataset)

    return f"The number of graphs in the dataset is: {graphs_nr}"


########################################################################################################################
# [0.3.] Get a selected graph in JSON format ===========================================================================
########################################################################################################################
@app.route('/select_graph', methods=['POST'])
def select_graph():
    """
    Select the graph by index

    :return:
    """

    graphs_nr = len(dataset)

    # Get the index of the selected graph ------------------------------------------------------------------------------
    req_data = request.get_json()
    select_graph_idx = req_data["select_graph_idx"]

    if 0 <= select_graph_idx < graphs_nr:
        graph_idx = select_graph_idx
        graph_json = graph_to_json(dataset[select_graph_idx])
        return graph_json
    else:
        return f"The selected graph with index: {select_graph_idx} " \
               f"is not compatible with the number of graphs in the dataset: {graphs_nr}"


########################################################################################################################
# [1.] Add node ========================================================================================================
########################################################################################################################
@app.route('/add_node_json', methods=['POST'])
def adding_node():
    """
    Add a new node and the JSON formatted part of its features
    """

    input_graph = dataset[graph_idx]

    # Get the features of the node -------------------------------------------------------------------------------------
    req_data = request.get_json()
    node_features = np.array(req_data["features"]).reshape(-1, 1).T

    # Add the node with its features -----------------------------------------------------------------------------------
    output_graph = add_node(input_graph, node_features)
    output_graph_json = graph_to_json(output_graph)

    return output_graph_json


########################################################################################################################
# [2.] Delete node =====================================================================================================
########################################################################################################################
@app.route('/graph_delete_node/<int:deleted_node_index>', methods=['DELETE'])
def delete_node(deleted_node_index: int):
    """
    Delete the node from the graph by index

    :param deleted_node_index: Deleted node index
    """

    input_graph = dataset[graph_idx]
    output_graph = remove_node(input_graph, deleted_node_index)
    output_graph_json = graph_to_json(output_graph)

    return output_graph_json


########################################################################################################################
# [3.] Add edge ========================================================================================================
########################################################################################################################
@app.route('/add_edge_json', methods=['POST'])
def adding_edge():
    """
    Add a new edge and the JSON formatted part of its features
    """

    input_graph = dataset[graph_idx]

    # Get the edge's "docking" points and its features -----------------------------------------------------------------
    req_data = request.get_json()

    new_edge_index_left = req_data["new_edge_index_left"]
    new_edge_index_right = req_data["new_edge_index_right"]
    edge_features = np.array(req_data["features"]).reshape(-1, 1).T

    # Add the node with its features -----------------------------------------------------------------------------------
    output_graph = add_edge(input_graph, new_edge_index_left, new_edge_index_right, edge_features)
    output_graph_json = graph_to_json(output_graph)

    return output_graph_json


########################################################################################################################
# [4.] Delete edge =====================================================================================================
########################################################################################################################
@app.route('/graph_delete_edge/<int:edge_index_left>/<int:edge_index_right>', methods=['DELETE'])
def delete_edge(edge_index_left: int, edge_index_right: int):
    """
    Delete the edge from the graph by indexes of the graph nodes that it connects

    :param edge_index_left: Index of left node of the edge
    :param edge_index_right: Index of right node of the edge
    """

    input_graph = dataset[graph_idx]
    output_graph = remove_edge(input_graph, edge_index_left, edge_index_right)
    output_graph_json = graph_to_json(output_graph)

    return output_graph_json


########################################################################################################################
# [5.] Add feature to all nodes ========================================================================================
########################################################################################################################
@app.route('/add_feature_to_all_nodes_json', methods=['POST'])
def add_feature_to_all_nodes():
    """
    Add a new feature to all nodes.
    Presupposes that the number and interpretation of node features is already known.
    """

    input_graph = dataset[graph_idx]

    # Get the new feature values for all the nodes ---------------------------------------------------------------------
    req_data = request.get_json()
    new_nodes_feature = np.array(req_data["new_nodes_feature"]).reshape(-1, 1)

    # Add the new feature in the graph ---------------------------------------------------------------------------------
    output_graph = add_feature_all_nodes(input_graph, new_nodes_feature)
    output_graph_json = graph_to_json(output_graph)

    return output_graph_json


########################################################################################################################
# [6.] Remove feature to all nodes =====================================================================================
########################################################################################################################
@app.route('/remove_feature_from_all_nodes_json', methods=['DELETE'])
def remove_feature_from_all_nodes():
    """
    Remove a feature from all nodes by index
    """

    input_graph = dataset[graph_idx]

    # Get the features of the node -------------------------------------------------------------------------------------
    req_data = request.get_json()
    removed_node_feature_idx = req_data["removed_nodes_feature_idx"]

    # Remove the feature from all nodes --------------------------------------------------------------------------------
    output_graph = remove_feature_all_nodes(input_graph, removed_node_feature_idx)
    output_graph_json = graph_to_json(output_graph)

    return output_graph_json


########################################################################################################################
# [7.] Add feature to all edges ========================================================================================
########################################################################################################################
@app.route('/add_feature_to_all_edges_json', methods=['POST'])
def add_feature_to_all_edges():
    """
    Add a new feature to all edges.
    Presupposes that the number and interpretation of edge features is already known.
    """

    input_graph = dataset[graph_idx]

    # Get the new feature values for all the nodes ---------------------------------------------------------------------
    req_data = request.get_json()
    new_node_feature = np.array(req_data["new_edges_feature"]).reshape(-1, 1)

    # Add the new feature in the graph ---------------------------------------------------------------------------------
    output_graph = add_feature_all_edges(input_graph, new_node_feature)
    output_graph_json = graph_to_json(output_graph)

    return output_graph_json


########################################################################################################################
# [8.] Remove feature to all edges =====================================================================================
########################################################################################################################
@app.route('/remove_feature_from_all_edges_json', methods=['DELETE'])
def remove_feature_from_all_edges():
    """
    Remove a feature from all edges by index.
    """

    input_graph = dataset[graph_idx]

    # Get the features of the edge -------------------------------------------------------------------------------------
    req_data = request.get_json()
    removed_edge_feature_idx = req_data["removed_edges_feature_idx"]

    # Remove the feature from all edges --------------------------------------------------------------------------------
    output_graph = remove_feature_all_edges(input_graph, removed_edge_feature_idx)
    output_graph_json = graph_to_json(output_graph)

    return output_graph_json


########################################################################################################################
# MAIN >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
########################################################################################################################
if __name__ == "__main__":
    app.run(debug=True)
