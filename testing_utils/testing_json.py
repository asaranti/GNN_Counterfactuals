"""
    JSON Testing

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-10-21
"""

import json
import numpy as np

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

print(f"Number of nodes: {input_graph.num_nodes}")
print(f"Number of edges: {input_graph.num_edges} \n")

########################################################################################################################
# [2.] JSONify the data ================================================================================================
########################################################################################################################
print(input_graph.x.cpu().numpy().shape)

input_graph_x = input_graph.x.cpu().tolist()
input_graph_y = input_graph.y.cpu().tolist()
input_graph_edge_attr = input_graph.edge_attr.cpu().tolist()
input_graph_edge_index = input_graph.edge_index.cpu().tolist()

print("------------------------------------------------------------------")
print("Edge indexes:")
print(input_graph_edge_index[0])
print(input_graph_edge_index[1])
print("------------------------------------------------------------------")
print("Edge attributes:")
print(np.array(input_graph_edge_attr).shape)
print("------------------------------------------------------------------")

graph_dict = {"x": input_graph_x,
              "y": input_graph_y,
              "edge_attr": input_graph_edge_attr,
              "edge_index": input_graph_edge_index}

graph_json = json.dumps(graph_dict)

print(graph_json)
