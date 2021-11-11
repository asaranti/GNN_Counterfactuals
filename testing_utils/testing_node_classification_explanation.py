"""
    Test the node classification and its explanation

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-11-11
"""

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from gnns.gnns_node_classification.node_classifier import node_classification

# [1.] Graphs dataset that was used in the GNN task --------------------------------------------------------------------
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

# [2.] Perform the first task (atm: node classification) with the GNN --------------------------------------------------
nodes_and_edges_relevances = node_classification(dataset)


