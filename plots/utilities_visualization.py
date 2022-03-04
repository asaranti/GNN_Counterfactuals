"""
    Utilities Visualization
    From Random_Kirc to Pytorch

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-02-17
"""

import matplotlib.pyplot as plt


def histogram_viz(cc_graphs_counter: dict):
    """
    Histogram visualization

    :param cc_graphs_counter: Connected Components histogram
    """

    fig = plt.figure(figsize=(12, 12))
    plt.bar(list(cc_graphs_counter.keys()), cc_graphs_counter.values(), color='b')
    plt.xlabel("Number of nodes", fontsize=26, fontweight='bold')
    plt.ylabel("Nr. of sub-graph components with this nr. of nodes", fontsize=26, fontweight='bold')
    plt.title("Nr. of sub-graph components with nr. of nodes", fontsize=32, fontweight='bold')
    plt.show()
    plt.close()
