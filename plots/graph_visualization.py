"""
    Graph visualization

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-10-05
"""

import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric


def graph_viz(graph_data: torch_geometric.data.data.Data):
    """
    Graph visualization with matplotlib

    :param graph_data: Graph data
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
    plt.show()
    plt.close()
