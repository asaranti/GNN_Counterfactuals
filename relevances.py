"""
    Presentation of relevances API

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-11-10
"""


from flask import Flask, request

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from gnns.gnns_node_classification.node_classifier import node_classification

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


########################################################################################################################
# [1.] Get the relevances for all nodes ================================================================================
########################################################################################################################
@relevances_app.route('/relevances_nodes', methods=['GET'])
def get_relevances_nodes():
    """
    Get the relevances of all nodes

    :return:
    """


########################################################################################################################
# [2.] Get the relevances for all edges ================================================================================
########################################################################################################################
@relevances_app.route('/relevances_edges', methods=['GET'])
def get_relevances_edges():
    """
    Get the relevances of all edges

    :return:
    """


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

    # [A.] Perform the first task (atm: node classification) with the GNN ----------------------------------------------
    #      Get the relevances for each node ----------------------------------------------------------------------------
    node_classification(dataset)

    relevances_app.run(debug=True)

