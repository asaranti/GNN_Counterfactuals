"""
    Format Transformation Experiment Nr. 3
    Transform the KIRC_RANDOM dataset to pytorch-compatible format

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-01-27
"""

import os

from preprocessing.format_transformations.format_transformation_random_kirc_to_pytorch import import_random_kirc_data

########################################################################################################################
# [1.] Transformation Experiment ::: From PPI to Pytorch_Graph =========================================================
########################################################################################################################

dataset_folder = os.path.join("data", "KIRC_RANDOM")
pytorch_random_kirc_edges_file = "KIDNEY_RANDOM_PPI.txt"
pytorch_random_kirc_mRNA_attribute_file = "KIDNEY_RANDOM_mRNA_FEATURES.txt"
pytorch_random_kirc_methy_attribute_file = "KIDNEY_RANDOM_Methy_FEATURES.txt"
pytorch_random_kirc_target_file = "KIDNEY_RANDOM_TARGET.txt"

protein_graph_list = import_random_kirc_data(dataset_folder,
                                             pytorch_random_kirc_mRNA_attribute_file,
                                             pytorch_random_kirc_methy_attribute_file,
                                             pytorch_random_kirc_edges_file,
                                             pytorch_random_kirc_target_file)


