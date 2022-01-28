"""
    Example code for the data structure that stores the (potentially) changed graphs

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-12-17
"""

from collections import OrderedDict
import copy
import re
import random

import numpy as np
import torch

from examples.synthetic_graph_examples.ba_graphs_generator import ba_graphs_gen

########################################################################################################################
# [1.] BA graphs generation ============================================================================================
########################################################################################################################
graphs_nr = 6
nodes_nr = 10
edges_per_node_nr = 2    # Number of edges to attach from a new node to existing nodes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

node_features_nr = 5
edge_features_nr = 4

graphs_list = ba_graphs_gen(graphs_nr, nodes_nr, edges_per_node_nr, node_features_nr, edge_features_nr)

########################################################################################################################
# [2.] Create 2-level dictionary for graph storage =====================================================================
########################################################################################################################
graph_adaptation_structure = {}
graph_id_composed_regex = "graph_id_[0-9]+_[0-9]+"

for graph in graphs_list:

    # 2.1. Use the graph_id to "position" the graph into the "graph_adaptation_structure" ------------------------------
    graph_id_composed = graph.graph_id
    pattern = re.compile(graph_id_composed_regex)
    graph_id_matches = bool(pattern.match(graph_id_composed))

    assert graph_id_matches, f"The graph's id {graph_id_composed} does not match " \
                             f"the required pattern: {graph_id_composed_regex}"

    # 2.2. Create the initial "graph_adaptation_structure" -------------------------------------------------------------
    graph_id_comp_array = graph_id_composed.split("_")
    patient_id = graph_id_comp_array[2]
    graph_id = graph_id_comp_array[3]

    patient_dict = OrderedDict()
    patient_dict[graph_id] = graph
    graph_adaptation_structure[patient_id] = patient_dict

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
print(f"Print all graphs before any changes")
print(graph_adaptation_structure)
print("=============================================================================================================\n")

########################################################################################################################
# [3.] Change some graph - remove the first edge =======================================================================
########################################################################################################################

# 3.1. Pick randomly a graph to change ---------------------------------------------------------------------------------
patient_idx_selected = str(random.randint(0, graphs_nr))
patient_graph_changes_nr = 0
patient_graph_selected = graph_adaptation_structure[patient_idx_selected][str(patient_graph_changes_nr)]

# 3.2. Make a deepcopy of the object, so that the original is not affected by any of the changes -----------------------
patient_graph_adapted = copy.deepcopy(patient_graph_selected)
patient_graph_changes_nr += 1       # !!! IMMEDIATELY !!! Increase this counter ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 3.3. Remove the first edge -------------------------------------------------------------------------------------------
patient_graph_adapted_edge_index = patient_graph_adapted.edge_index.detach().cpu().numpy()
patient_graph_adapted.edge_index = torch.tensor(np.delete(patient_graph_adapted_edge_index, 0, 1))

patient_graph_adapted_edge_attr = patient_graph_adapted.edge_attr.detach().cpu().numpy()
patient_graph_adapted.edge_attr = torch.tensor(np.delete(patient_graph_adapted_edge_attr, 0, 0))

patient_graph_adapted_edge_ids = patient_graph_adapted.edge_ids
patient_graph_adapted.edge_ids = np.delete(patient_graph_adapted_edge_ids, 0)

########################################################################################################################
# [4.] Insert the graph into the dictionary ============================================================================
########################################################################################################################
# 3.4. Insert the graph into the dictionary ----------------------------------------------------------------------------
graph_adaptation_structure[patient_idx_selected][str(patient_graph_changes_nr)] = patient_graph_adapted

print(f"Print all graphs of the selected patient: {patient_idx_selected} (Original and adapted)")
print(graph_adaptation_structure[patient_idx_selected])
print("---------------------------------------------------------------------------------------------------------------")
print(f"Print all graphs after the changes")
print(graph_adaptation_structure)
print("=============================================================================================================\n")
