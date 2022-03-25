"""
    GNN explanations with Captum

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-03-04
"""

from collections import defaultdict
import numpy as np

from captum.attr import Saliency, IntegratedGradients
import torch
from torch_geometric.data.data import Data

from gnns.gnns_graph_classification.GCN_Graph_Classification import GCN


def model_forward(edge_mask: torch.Tensor, model: GCN, data: Data, device: int):
    """
    Forward the data with the added mask "on top" in the GCN

    :param edge_mask:
    :param model:
    :param data:
    :param device:
    """

    batch = torch.zeros(data.x.shape[0], dtype=int).to(device)

    out = model(data.x, data.edge_index, batch, edge_mask)
    return out


def explain(method: str, model: GCN, data: Data, device: str, target_label: int):
    """

    :param method: Explanation method
    :param model: GNN model that will be used for the explanation
    :param data: Input data
    :param target_label: Target label
    :param device: Device (Cuda or cpu)
    """

    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)

    if method == 'ig':
        ig = IntegratedGradients(model_forward)
        mask = ig.attribute(input_mask, target=target_label,
                            additional_forward_args=(model, data, device),
                            internal_batch_size=data.edge_index.shape[1])
    elif method == 'saliency':
        saliency = Saliency(model_forward)
        mask = saliency.attribute(input_mask, target=target_label,
                                  additional_forward_args=(model, data, device))
    else:
        raise Exception('Unknown explanation method')

    edge_mask = np.abs(mask.cpu().detach().numpy())
    if edge_mask.max() > 0:                         # avoid division by zero
        edge_mask = edge_mask / edge_mask.max()
    return edge_mask


def aggregate_edge_directions(edge_mask, data):
    """
    Aggregate for all edge directions

    :param edge_mask:
    :param data:
    """

    print(type(edge_mask), type(data))

    edge_mask_dict = defaultdict(float)

    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val

    return edge_mask_dict


def explain_sample(method: str, model: GCN, data: Data, target_label: int, device: str) -> list:
    """
    Explain input sample

    :param method: Explanation method
    :param model: GNN model that will be used for the explanation
    :param data: Input data
    :param target_label: Target label
    :param device: Device (Cuda or cpu)

    :return: List of edge relevances
    """

    edge_mask = explain(method, model, data, device, target_label)

    return edge_mask
