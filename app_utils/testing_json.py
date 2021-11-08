"""
    JSON Testing

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-10-21
"""

import json

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

########################################################################################################################
# [1.] Import data, split them to training and test datasets ===========================================================
########################################################################################################################
dataset = TUDataset(root='data/TUDataset', name='MUTAG')

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

input_graph = dataset[0]  # Get the first graph object ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print()
print(type(input_graph))

########################################################################################################################
# [2.] JSONify the data ================================================================================================
########################################################################################################################
input_graph_x = input_graph.x.cpu().tolist()
input_graph_y = input_graph.y.cpu().tolist()
input_graph_edge_attr = input_graph.edge_attr.cpu().tolist()
input_graph_edge_index = input_graph.edge_index.cpu().tolist()

graph_dict = {"x": input_graph_x,
              "y": input_graph_y,
              "edge_attr": input_graph_edge_attr,
              "edge_index": input_graph_edge_index}

graph_json = json.dumps(graph_dict)

print(graph_json)
