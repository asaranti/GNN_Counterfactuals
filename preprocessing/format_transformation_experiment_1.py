"""
    Format Transformation Experiment Nr. 2

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-12-15
"""

import os

from preprocessing.format_transformation_ppi_to_pytorch import transform_from_ppi_to_pytorch
from preprocessing.format_transformation_ui_to_pytorch import transform_from_pytorch_to_ui

########################################################################################################################
# [1.] Transformation Experiment ::: From PPI to Pytorch_Graph =========================================================
########################################################################################################################

dataset_folder = os.path.join("data", "Protein_Dataset")
pytorch_ppi_attributes_file = "Human__TCGA_ACC__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct"
pytorch_ppi_node_id_to_name_file = "human.name_2_string.csv"
pytorch_ppi_edges_file = "9606.protein.links.v11.0.txt"

ppi_pytorch_graph = transform_from_ppi_to_pytorch(dataset_folder,
                                                  pytorch_ppi_attributes_file,
                                                  pytorch_ppi_node_id_to_name_file,
                                                  pytorch_ppi_edges_file)
print(ppi_pytorch_graph)

########################################################################################################################
# [2.] Transformation Experiment ::: From Pytorch_Graph to UI Format ===================================================
########################################################################################################################
ui_pytorch_nodes_file_protein = "ppi_nodes_ui_format.csv"
ui_pytorch_edges_file_protein = "ppi_edges_ui_format.csv"
transform_from_pytorch_to_ui(ppi_pytorch_graph,
                             dataset_folder,
                             ui_pytorch_nodes_file_protein,
                             ui_pytorch_edges_file_protein)
