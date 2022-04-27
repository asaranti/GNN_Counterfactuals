"""
    Graph visualization

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-10-05
"""

import os

from bokeh.io import output_file, show
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          MultiLine, ResetTool)
from bokeh.plotting import figure, from_networkx

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import torch_geometric


def graph_visualization_complex(graph_data: torch_geometric.data.data.Data, graph_idx: int):
    """
    More detailed visualization of the graph

    :param graph_data: Graph data
    :param graph_idx: Index of graph
    :return:
    """

    # [1.] Generation of figure ----------------------------------------------------------------------------------------
    p_figure = figure(width=1200, height=1200)
    p_figure.title.text = f"Graph nr. {graph_idx}"
    p_figure.title.text_font_size = '24pt'

    # [2.] Plot the graph ----------------------------------------------------------------------------------------------
    graph_nx = to_networkx(graph_data)
    graph_renderer = from_networkx(graph_nx, nx.spring_layout, center=(0, 0))
    p_figure.renderers.append(graph_renderer)

    # [3.] Extra tools (hover...) --------------------------------------------------------------------------------------
    node_hover_tool = HoverTool(tooltips=[("node_id", "@node_id")], renderers=[graph_renderer.node_renderer])
    p_figure.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())

    # [4.] Transform the network for HTML output -----------------------------------------------------------------------
    output_data_path = os.path.join(os.path.join("data", "output", "KIRC_RANDOM", "plots", "graph_plots"))
    output_file(os.path.join(output_data_path, f"graph_{graph_idx}.html"))
    show(p_figure)


def graph_viz(graph_data: torch_geometric.data.data.Data, graph_idx: int):
    """
    Simple graph visualization with matplotlib
    Not so many details in the plot

    :param graph_data: Graph data
    :param graph_idx: Graph index
    :return:
    """

    graph_viz = torch_geometric.utils.to_networkx(graph_data, to_undirected=True)

    fig = plt.figure(figsize=(12, 12))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(
        graph_viz,
        with_labels=True,
    )
    # plt.show()

    output_data_path = os.path.join(os.path.join("data", "output", "KIRC_RANDOM", "plots", "graph_plots"))
    fig.savefig(os.path.join(output_data_path,  f"graph_{graph_idx}.png"))

    plt.close()


def graph_viz_general(graph_data: torch_geometric.data.data.Data):
    """
    The most general graph visualisation with matplotlib

    :param graph_data: Graph data
    """

    graph_viz = torch_geometric.utils.to_networkx(graph_data, to_undirected=True)

    fig = plt.figure(figsize=(12, 12))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(
        graph_viz,
        with_labels=True,
    )
    plt.show()

    plt.close()
    