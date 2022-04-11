"""
    Data format consistency

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2022-04-11
"""

import os
import pickle
import random

from constraints.graph_constraints import check_data_format_consistency


########################################################################################################################
# MAIN Test ============================================================================================================
########################################################################################################################
def test_data_format_consistency():
    """
    Test data format consistency
    """

    # [1.] Transformation Experiment ::: From PPI to Pytorch_Graph -----------------------------------------------------
    dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
    dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))

    dataset_len = len(dataset)
    graph_idx = random.randint(0, dataset_len)
    input_graph = dataset[graph_idx]

    # [2.] Check the data format consistency ---------------------------------------------------------------------------
    check_data_format_consistency(input_graph)

