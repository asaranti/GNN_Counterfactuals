"""
    Graph explanations visualization

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-03-17
"""

import os
import shutil

from bokeh.io import output_file, show
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          MultiLine, ResetTool)
from bokeh.plotting import figure, from_networkx
import networkx as nx
import torch
import torch_geometric
from torch_geometric.utils.convert import to_networkx


def integrated_gradients_viz(graph_data: torch_geometric.data.data.Data,
                             graph_idx: int,
                             edge_mask: list,
                             node_labels_list: list,
                             ground_truth_label: int,
                             prediction_label: int,
                             explanation_label: int,
                             output_data_path: str):
    """
    Integrated gradients

    :param graph_data: Graph data
    :param graph_idx: Index of graph
    :param edge_mask: Edge relevance values
    :param node_labels_list: List of node labels
    :param ground_truth_label: Ground truth label
    :param prediction_label: Prediction label
    :param explanation_label: Label w.r.t. which the explanation was computed
    :param output_data_path: Output data path for the plots
    :return:
    """

    # [1.] Generation of figure ----------------------------------------------------------------------------------------
    p_figure = figure(width=1200, height=1200)
    p_figure.title.text = f"Graph nr. {graph_idx}, Ground truth: {ground_truth_label}, Predicted: {prediction_label}," \
                          f"Explanation: {explanation_label}"
    p_figure.title.text_font_size = '24pt'

    # [2.] Plot the graph ----------------------------------------------------------------------------------------------
    graph_nx = nx.Graph()
    for node_idx in range(len(node_labels_list)):
        graph_nx.add_node(node_idx, node_label=node_labels_list[node_idx])

    graph_nx_edges = list(graph_nx.edges)
    graph_nx.remove_edges_from(graph_nx_edges)

    graph_renderer = from_networkx(graph_nx, nx.spring_layout, center=(0, 0))
    graph_renderer.node_renderer.glyph = Circle(size=5, fill_color='black')
    graph_renderer.edge_renderer.glyph = MultiLine(line_color="black", line_alpha=0.8, line_width=1)
    p_figure.renderers.append(graph_renderer)

    # [3.] Extra tools (hover...) --------------------------------------------------------------------------------------
    node_hover_tool = HoverTool(tooltips=[("index", "@index"), ("node_label", "@node_label")],
                                renderers=[graph_renderer.node_renderer])

    # [4.] Add the relevance of the edges "on top" ---------------------------------------------------------------------
    layout_pos = graph_renderer.layout_provider.graph_layout
    edge_index_array = graph_data.edge_index.cpu().detach().numpy()
    p_lines_all = []
    for edge_cnt in range(edge_index_array.shape[1]):

        edge = edge_index_array[:, edge_cnt]

        edge_left = edge[0]
        edge_right = edge[1]

        nodes_x_coors = [layout_pos[edge_left][0], layout_pos[edge_right][0]]
        nodes_y_coors = [layout_pos[edge_left][1], layout_pos[edge_right][1]]

        relevance_of_line = edge_mask[edge_cnt]

        p_line = p_figure.line(
            nodes_x_coors,
            nodes_y_coors,
            line_width=2,
            color='red',
            alpha=float(relevance_of_line)
        )
        p_lines_all.append(p_line)

    edge_hover_tool = HoverTool(tooltips="alpha: @alpha", renderers=p_lines_all, mode="mouse")
    p_figure.add_tools(node_hover_tool,
                       # edge_hover_tool,
                       BoxZoomTool(),
                       ResetTool())

    # [5.] Transform the network for HTML output -----------------------------------------------------------------------
    output_file(os.path.join(output_data_path, f"integrated_gradients_graph_{graph_idx}"
                                               f"_towards_{explanation_label}.html"))
    show(p_figure)

