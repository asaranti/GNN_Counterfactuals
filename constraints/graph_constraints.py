"""
    Graph constraints

    :author: Anna Saranti
    :copyright: © 2022 HCI-KDD (ex-AI) group
    :date: 2022-04-11
"""

import torch
import torch_geometric

import numpy as np


def check_data_format_consistency(input_graph: torch_geometric.data.data.Data):
    """
    Check the data format consistency, that the types of the elements are as expected.
    Those types should not be changed, although it is allowed to add new

    :param input_graph: Input graph
    """

    assert isinstance(input_graph.x, torch.Tensor)
    assert isinstance(input_graph.edge_index, torch.Tensor)
    assert isinstance(input_graph.edge_attr, torch.Tensor) or input_graph.edge_attr is None
    assert isinstance(input_graph.y, torch.Tensor)
    assert isinstance(input_graph.node_labels, list)
    assert isinstance(input_graph.node_ids, list)
    assert isinstance(input_graph.node_feature_labels, list)
    assert isinstance(input_graph.edge_ids, list)

    if hasattr(input_graph, 'edge_attr_labels'):
        assert isinstance(input_graph.edge_attr_labels, list) or input_graph.edge_attr is None

    assert isinstance(input_graph.pos, torch.Tensor) or input_graph.pos is None
    assert isinstance(input_graph.graph_id, str)
