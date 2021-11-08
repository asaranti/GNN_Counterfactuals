"""
    Flask application instance

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-10-18
"""

from flask import Flask, jsonify

from torch_geometric.datasets import TUDataset

from actionable.graph_actions import remove_node, remove_edge
from app_utils.jsonification import graph_to_json

########################################################################################################################
app = Flask(__name__)


# Start: Index ---------------------------------------------------------------------------------------------------------
@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"


dataset = TUDataset(root='data/TUDataset', name='MUTAG')
graph_idx = 0


# [1.] Get the first graph ---------------------------------------------------------------------------------------------
@app.route('/graph', methods=['GET'])
def get():
    """
    Get the first graph

    :return:
    """

    graph_json = graph_to_json(dataset[graph_idx])

    return graph_json


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
# MAIN =================================================================================================================
########################################################################################################################
if __name__ == "__main__":
    app.run(debug=True)
