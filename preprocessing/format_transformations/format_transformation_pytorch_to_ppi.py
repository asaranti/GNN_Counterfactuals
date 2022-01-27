"""
    Format Transformation
    From Pytorch to PPI

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-12-01
"""

from math import log
import os

import numpy as np
import pandas as pd
from torch_geometric.data import Data


def transform_from_pytorch_to_ppi(graph: Data,
                                  input_dataset_folder: str,
                                  pytorch_to_ppi_attributes_file: str,
                                  pytorch_to_ppi_node_id_to_name_file: str,
                                  pytorch_to_ppi_edges_file: str):
    """
    Apply the transformation between PPI format to the Pytorch format

    :param graph: Input graph
    :param input_dataset_folder: Dataset folder where the files will be placed
    :param pytorch_to_ppi_attributes_file: Attributes file that will be written
    :param pytorch_to_ppi_node_id_to_name_file: File containing the mapping from node ids to node names
    :param pytorch_to_ppi_edges_file: File containing the edge connections
    """

    ####################################################################################################################
    # [1.] Import Pytorch dataset and transform the attributes =========================================================
    ####################################################################################################################
    attributes = graph.x.numpy()
    attributes_shape = attributes.shape
    nodes_nr = attributes_shape[0]
    features_nr = attributes_shape[1]

    feature_names_list = ['feature_id_' + str(x) for x in range(features_nr)]
    nodes_ids_list = ['node_id_' + str(x) for x in range(nodes_nr)]

    attributes_data_dict = {'attrib_name': nodes_ids_list}
    for feature_nr in range(features_nr):
        attributes_data_dict['feature_id_' + str(feature_nr)] = list(attributes[:, feature_nr])
    attributes_data_df = pd.DataFrame.from_dict(attributes_data_dict)
    # print(attributes_data_df.head(5))

    attributes_data_df.to_csv(os.path.join(input_dataset_folder, pytorch_to_ppi_attributes_file),
                              index=False, sep='\t')

    ####################################################################################################################
    # [2.] Create a file with the mapping of "node_id" to "node_name" ==================================================
    ####################################################################################################################
    nodes_names_list = ['node_name_' + str(x) for x in range(nodes_nr)]
    # random.shuffle(nodes_names_list)

    first_col_list = ['9606' for x in range(nodes_nr)]
    node_id_names_mapping_dict = {'first_col': first_col_list,
                                  'node_id': nodes_ids_list,
                                  'node_name': nodes_names_list}
    node_id_names_mapping_df = pd.DataFrame.from_dict(node_id_names_mapping_dict)

    node_id_to_name_file = os.path.join(input_dataset_folder, pytorch_to_ppi_node_id_to_name_file)
    node_id_to_name_header = '# NCBI taxid / display name / STRING 		\n'
    with open(node_id_to_name_file, 'w') as fp:
        fp.write(node_id_to_name_header)
        fp.close()

    node_id_names_mapping_df.to_csv(node_id_to_name_file,
                                    index=False, sep='\t', header=False, mode='a')

    ####################################################################################################################
    # [3.] Create the edges file =======================================================================================
    ####################################################################################################################
    edges_file = os.path.join(input_dataset_folder, pytorch_to_ppi_edges_file)

    max_relevance_val = 100

    edge_indexes = graph.edge_index.numpy()
    edge_names_list = []
    for col_idx in range(edge_indexes.shape[1]):

        edge_ids = edge_indexes[:, col_idx]
        edge_names_list.append([nodes_names_list[edge_ids[0]],
                                nodes_names_list[edge_ids[1]],
                                # str(random.randint(0, max_relevance_val))
                                str(log(col_idx + 1))
                                ])

    edge_data_array = np.array(edge_names_list)

    edge_data_dict = {'protein1': edge_data_array[:, 0],
                      'protein2': edge_data_array[:, 1],
                      'combined_score': edge_data_array[:, 2]}
    edge_data_df = pd.DataFrame.from_dict(edge_data_dict)
    # print(edge_data_df.head(5))

    edge_data_df.to_csv(edges_file, index=False, sep=' ')

"""
########################################################################################################################
# MAIN: Perform format transformations =================================================================================
########################################################################################################################
pytorch_ppi_attributes_file = "pytorch_to_ppi_attributes.cct"
pytorch_ppi_node_id_to_name_file = "pytorch_to_ppi_node_id_to_name.csv"
pytorch_ppi_edges_file = "pytorch_to_ppi_edges.txt"

# [1.] Pytorch to ppi format ===========================================================================================
dataset = Planetoid(root='/tmp/Cora', name='Cora')
original_graph = dataset[0]
print(original_graph)
transform_from_pytorch_to_ppi(original_graph,
                              os.path.join("data", "format_transformation_a"),
                              pytorch_ppi_attributes_file,
                              pytorch_ppi_node_id_to_name_file,
                              pytorch_ppi_edges_file)

# [2.] Re-import this to Pytorch format ================================================================================
reimported_graph = transform_from_ppi_to_pytorch(os.path.join("data", "format_transformation_a"),
                                                 pytorch_ppi_attributes_file,
                                                 pytorch_ppi_node_id_to_name_file,
                                                 pytorch_ppi_edges_file)
print(f"Reimported graph: {reimported_graph}")

# [3.] Pytorch to ppi Nr.2 =============================================================================================
transform_from_pytorch_to_ppi(reimported_graph,
                              os.path.join("data", "format_transformation_b"),
                              pytorch_ppi_attributes_file,
                              pytorch_ppi_node_id_to_name_file,
                              pytorch_ppi_edges_file)

# [4.] Compare the resulted files ======================================================================================
pytorch_to_ppi_cmp_1 = filecmp.cmp(os.path.join("data", "format_transformation_a", pytorch_ppi_attributes_file),
                                   os.path.join("data", "format_transformation_b", pytorch_ppi_attributes_file))

pytorch_to_ppi_cmp_2 = filecmp.cmp(os.path.join("data", "format_transformation_a", pytorch_ppi_node_id_to_name_file),
                                   os.path.join("data", "format_transformation_b", pytorch_ppi_node_id_to_name_file))

pytorch_to_ppi_cmp_3 = filecmp.cmp(os.path.join("data", "format_transformation_a", pytorch_ppi_edges_file),
                                   os.path.join("data", "format_transformation_b", pytorch_ppi_edges_file))

print(pytorch_to_ppi_cmp_1, pytorch_to_ppi_cmp_2, pytorch_to_ppi_cmp_3)
"""
