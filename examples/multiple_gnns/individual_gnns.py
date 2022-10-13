"""
    Individual GNNs for each of the datasets

    :author: Anna Saranti
    :copyright: © 2022 HCI-KDD (ex-AI) group
    :date: 2022-10-12
"""

import os
import pickle
import random

from actionable.gnn_actions import GNN_Actions
from actionable.gnn_explanations import explain_sample
from actionable.graph_actions import add_node, remove_node
from gnns.gnn_selectors.gnn_definitions import define_gnn

# [0.] -----------------------------------------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

# [1.] Select dataset --------------------------------------------------------------------------------------------------
# [1.a.] KIRC Subnet ---------------------------------------------------------------------------------------------------
# dataset_name = "kirc_subnet"
# dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")

# [1.b.] KIRC random nodes ui ------------------------------------------------------------------------------------------
dataset_name = "kirc_random_nodes_ui"
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")

# [1.c.] Synthetic -----------------------------------------------------------------------------------------------------
# dataset_name = "synthetic"
# dataset_pytorch_folder = os.path.join("data", "output", "Synthetic", "synthetic_pytorch")


dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, f'{dataset_name}_pytorch.pkl'), "rb"))

# [2.] Select GNN architecture -----------------------------------------------------------------------------------------
gnn_architecture_params_dict = define_gnn(dataset_name)
print(gnn_architecture_params_dict)

# [3.] Train the GNN for the first time --------------------------------------------------------------------------------
gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, dataset_name)
model, performance_values_dict = gnn_actions_obj.gnn_init_train(dataset)
print(performance_values_dict)

# [4.] Delete one node -------------------------------------------------------------------------------------------------
dataset_len = len(dataset)
graph_idx = random.randint(0, dataset_len - 1)
input_graph = dataset[graph_idx]
print(input_graph)

nodes_orig_nr = input_graph.x.shape[0]
print(f"Nr. of nodes original: {nodes_orig_nr}")

node_idx = 0
output_graph = remove_node(input_graph, node_idx)
nodes_output_nr = output_graph.x.shape[0]
print(f"Nr. of nodes after node delete: {nodes_output_nr}")

dataset[graph_idx] = output_graph

# [5.] Explanation -----------------------------------------------------------------------------------------------------
explanation_method = 'gnnexplainer'     # Also possible: 'ig' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ground_truth_label = int(input_graph.y.cpu().detach().numpy()[0])
explanation_label = ground_truth_label  # Can also be the opposite - all possible combinations of 0 and 1 ~~~~~~~~~~~~~~

# GNNECPLAINER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
node_mask = explain_sample(explanation_method, model, input_graph, explanation_label)
print(f"\nGNNExplainer mask: {node_mask}")


# CAPTUM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
rel_pos = list(explain_sample(
        explanation_method,
        model,
        input_graph,
        explanation_label,
    ))
rel_pos = [str(round(edge_relevance, 2)) for edge_relevance in rel_pos]

print(rel_pos)
print(f"Captum relevances: {rel_pos}")
print(type(rel_pos[0]))

# [6.] Retrain ---------------------------------------------------------------------------------------------------------
model, performance_values_dict = gnn_actions_obj.gnn_retrain(dataset)
print(performance_values_dict)



