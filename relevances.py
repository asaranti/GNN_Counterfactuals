"""
    Presentation of relevances API

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-11-10
"""

import json

from flask import Flask, request

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from gnns.gnns_node_classification.node_classifier import GNNNodeClassifierExplainer

########################################################################################################################
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
########################################################################################################################
relevances_app = Flask(__name__)


# Start: Index ---------------------------------------------------------------------------------------------------------
@relevances_app.route('/')
@relevances_app.route('/index')
def index():
    return "Hello Relevances!"


# Graphs dataset that was used in the GNN task -------------------------------------------------------------------------
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
graph_index = 0


########################################################################################################################
# [1.] Get the relevances for all nodes ================================================================================
########################################################################################################################
@relevances_app.route('/relevances_nodes', methods=['POST'])
def get_relevances_nodes():
    """
    Get the relevances of all nodes

    :return:
    """

    # [1.] User specifies the node to explain --------------------------------------------------------------------------
    req_data = request.get_json()
    node_to_explain = req_data["node_to_explain"]

    if not str(node_to_explain).isdigit():
        return f"The node index must be a valid integer. Instead it is: {node_to_explain}"

    selected_graph = dataset[graph_index]
    if 0 <= node_to_explain < selected_graph.num_nodes:

        # [2.] Compute and return relevances ---------------------------------------------------------------------------
        nodes_and_edges_relevances = gnn_node_classification_explainer.node_explanation(dataset, node_to_explain)
        return json.dumps(nodes_and_edges_relevances["graph_nodes_relevances"])

    else:
        return f"The node index: {node_to_explain} is not compatible with the number of nodes " \
               f"in the graph: {selected_graph.num_nodes}"


########################################################################################################################
# [2.] Get the relevances for all edges ================================================================================
########################################################################################################################
@relevances_app.route('/relevances_edges', methods=['POST'])
def get_relevances_edges():
    """
    Get the relevances of all edges

    :return:
    """

    # [1.] User specifies the node to explain --------------------------------------------------------------------------
    req_data = request.get_json()
    node_to_explain = req_data["node_to_explain"]

    if not str(node_to_explain).isdigit():
        return f"The node index must be a valid integer. Instead it is: {node_to_explain}"

    selected_graph = dataset[graph_index]
    if 0 <= node_to_explain < selected_graph.num_nodes:

        # [2.] Compute and return relevances ---------------------------------------------------------------------------
        nodes_and_edges_relevances = gnn_node_classification_explainer.node_explanation(dataset, node_to_explain)
        return json.dumps(nodes_and_edges_relevances["graph_edges_relevances"])
    else:
        return f"The node index: {node_to_explain} is not compatible with the number of nodes " \
               f"in the graph: {selected_graph.num_nodes}"


########################################################################################################################
# [3.] Get the relevances for all node features of the selected node ===================================================
########################################################################################################################
@relevances_app.route('/node_feature_relevances', methods=['POST'])
def get_node_feature_relevances():
    """
    Get the relevances of all features of the node

    :return:
    """


########################################################################################################################
# [4.] Get the relevances for all edge features of the selected edge ===================================================
########################################################################################################################
@relevances_app.route('/edge_feature_relevances', methods=['POST'])
def get_edge_feature_relevances():
    """
    Get the relevances of all features of the edge

    :return:
    """


########################################################################################################################
# MAIN >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
########################################################################################################################
if __name__ == "__main__":

    # [1.] Perform the first task (atm: node classification) with the GNN ----------------------------------------------
    # Get the relevances for each node and edge ------------------------------------------------------------------------
    graph_index = 0
    gnn_node_classification_explainer = GNNNodeClassifierExplainer()
    gnn_node_classification_explainer.node_classification(dataset, graph_index)

    relevances_app.run(debug=True)

