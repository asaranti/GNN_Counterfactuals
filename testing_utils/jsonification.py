"""
    Transform various objects to JSON

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-11-08
"""

import json

from torch_geometric.data.data import Data


def graph_to_json(input_graph: Data):
    """
    Input graph to JSON

    :param input_graph: Input graph
    """

    input_graph_x = input_graph.x.cpu().tolist()
    input_graph_y = input_graph.y.cpu().tolist()
    input_graph_edge_attr = input_graph.edge_attr.cpu().tolist()
    input_graph_edge_index = input_graph.edge_index.cpu().tolist()

    graph_dict = {"x": input_graph_x,
                  "y": input_graph_y,
                  "edge_attr": input_graph_edge_attr,
                  "edge_index": input_graph_edge_index}

    graph_json = json.dumps(graph_dict)

    return graph_json
