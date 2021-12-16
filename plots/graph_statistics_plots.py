"""
    Format Transformation
    From PPI to Pytorch

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-12-16
"""

import numpy as np
import matplotlib.pyplot as plt


def statistics_histogram(x_values: list, title_str: str):
    """
    Statistics histogram

    :param x_values: List of all values that will be plotted
    :param title_str: Description of the metric that will be plotted
    """

    n, bins, patches = plt.hist(x_values, 20, density=True, facecolor='b', alpha=0.75)

    plt.xlabel('Smarts')
    plt.ylabel('Distribution')
    plt.title('Histogram of ' + title_str)
    plt.grid(True)
    plt.show()
    plt.close()
