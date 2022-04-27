"""
    Format Transformation Experiment Nr. 3
    Transform the KIRC_RANDOM dataset to pytorch-compatible format

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-01-27
"""

import os
import pickle

import torch

from plots.graph_visualization import graph_visualization_complex, graph_viz
from preprocessing_files.format_transformations.format_transformation_random_kirc_to_pytorch import import_random_kirc_data
from preprocessing_files.format_transformations.format_transformation_pytorch_to_ui import transform_from_pytorch_to_ui

########################################################################################################################
# [1.] Transformation Experiment ::: From PPI to Pytorch_Graph =========================================================
########################################################################################################################

dataset_folder = os.path.join("data", "input", "KIRC_RANDOM", "kirc_random_orig")
pytorch_random_kirc_edges_file = "KIDNEY_RANDOM_PPI.txt"
pytorch_random_kirc_mRNA_attribute_file = "KIDNEY_RANDOM_mRNA_FEATURES.txt"
pytorch_random_kirc_methy_attribute_file = "KIDNEY_RANDOM_Methy_FEATURES.txt"
pytorch_random_kirc_target_file = "KIDNEY_RANDOM_TARGET.txt"

protein_graph_list = import_random_kirc_data(dataset_folder,
                                             pytorch_random_kirc_mRNA_attribute_file,
                                             pytorch_random_kirc_methy_attribute_file,
                                             pytorch_random_kirc_edges_file,
                                             pytorch_random_kirc_target_file)

########################################################################################################################
# [2.] Visualization of the dataset ====================================================================================
########################################################################################################################
print("Start the plots")

for graph_idx in range(len(protein_graph_list)):

    print(f"Graph index: {graph_idx}")
    graph = protein_graph_list[graph_idx]

    # [2.1.] Check that the topology of all graphs is equal ------------------------------------------------------------
    if graph_idx == 0:
        graph_edge_index_0 = graph.edge_index
    else:
        b_topologie_equal = torch.equal(graph_edge_index_0, graph.edge_index)
        assert b_topologie_equal, f"Is the topology equal? {b_topologie_equal}"

    # [2.2.] Check that the edges are not double there -----------------------------------------------------------------
    all_edges = graph.edge_index.detach().cpu().numpy()
    columns_nr = all_edges.shape[1]

    all_edges_sorted_list = []
    all_edges_sorted_set = set()
    for column_index in range(columns_nr):

        edge_component = list(all_edges[:, column_index])
        edge_component.sort()

        all_edges_sorted_list.append(edge_component)
        all_edges_sorted_set.add(tuple(edge_component))

    if len(all_edges_sorted_list) != len(all_edges_sorted_set):
        print("Non unique edges")

    # [2.3.] Actual visualization --------------------------------------------------------------------------------------
    # graph_viz(graph, graph_idx)
    # graph_visualization_complex(graph, graph_idx)

########################################################################################################################
# [3.] Store the Pytorch Dataset =======================================================================================
########################################################################################################################
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
with open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), 'wb') as f:
    pickle.dump(protein_graph_list, f)

########################################################################################################################
# [4.] From Pytorch_Graph to UI Format =================================================================================
########################################################################################################################
dataset_ui_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_ui")

graphs_nr = len(protein_graph_list)
graph_idx = 0
for graph_idx in range(graphs_nr):

    transform_from_pytorch_to_ui(protein_graph_list[graph_idx],
                                 dataset_ui_folder,
                                 f"kirc_random_nodes_ui_format_{graph_idx}.csv",
                                 f"kirc_random_egdes_ui_format_{graph_idx}.csv")

    graph_idx += 1

