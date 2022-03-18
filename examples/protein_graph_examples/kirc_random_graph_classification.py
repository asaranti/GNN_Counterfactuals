"""
    Graph classification of KIRC RANDOM dataset #TUDataset

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-01-27
"""

from datetime import datetime
import os
import pickle
import random
import shutil

import networkx as nx
import numpy as np
from operator import itemgetter
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import minmax_scale

from actionable.gnn_explanations import explain_sample
from gnns.gnns_graph_classification.gnn_train_test_methods import train, test
from gnns.gnns_graph_classification.GCN_Graph_Classification import GCN
from preprocessing.format_transformations.format_transformation_random_kirc_to_pytorch import import_random_kirc_data
from sklearn.preprocessing import StandardScaler
from plots.graph_explanations_visualization import integrated_gradients_viz

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

########################################################################################################################
# [1.] Transformation Experiment ::: From PPI to Pytorch_Graph =========================================================
########################################################################################################################
dataset_pytorch_folder = os.path.join("data", "output", "KIRC_RANDOM", "kirc_random_pytorch")
dataset = pickle.load(open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), "rb"))

########################################################################################################################
# [2.] Data Preparation ================================================================================================
########################################################################################################################

# [2.1.] Input features preprocessing/normalization --------------------------------------------------------------------
for graph in dataset:

    x_features = graph.x
    x_features_array = x_features.cpu().detach().numpy()

    # scaler = StandardScaler()
    # scaler.fit(x_features_array)
    # x_features_transformed = scaler.transform(x_features_array)
    x_features_transformed = minmax_scale(x_features_array, feature_range=(0, 1))
    graph.x = torch.tensor(x_features_transformed)
    graph.to(device)

# [2.2.] Split training/validation/test set ----------------------------------------------------------------------------
graph_0 = dataset[0]
num_features = graph_0.num_node_features
graphs_nr = len(dataset)

# [2.3.] Shuffle the dataset and keep the list indexes -----------------------------------------------------------------
x = list(enumerate(dataset))
random.shuffle(x)
random_indices, graphs_list = zip(*x)
dataset_random_shuffling = list(itemgetter(*random_indices)(dataset))

# [2.4.] Split to training and test set --------------------------------------------------------------------------------
train_dataset_len = int(graphs_nr*3/4)
train_dataset = dataset_random_shuffling[:train_dataset_len]
test_dataset = dataset_random_shuffling[train_dataset_len:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

########################################################################################################################
# [3.] Graph Classification ============================================================================================
########################################################################################################################
num_classes = 2
model = GCN(num_node_features=num_features, hidden_channels=200, num_classes=num_classes).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss().to(device)

# Training for some epochs ---------------------------------------------------------------------------------------------
date_time_obj = datetime.now()
time_stamp_srt = date_time_obj.strftime("%d-%b-%Y %H:%M:%S")
print(f'Training time start: {time_stamp_srt}')

epochs_nr = 20
for epoch in range(1, epochs_nr + 1):

    train(model, train_loader, optimizer, criterion)
    train_acc = test(model, train_loader)
    test_acc = test(model, test_loader)

    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    print("-------------------------------------------------------------------------")

date_time_obj = datetime.now()
time_stamp_srt = date_time_obj.strftime("%d-%b-%Y %H:%M:%S")
print(f'Training time end: {time_stamp_srt}')

test_acc = test(model, test_loader)

date_time_obj = datetime.now()
time_stamp_srt = date_time_obj.strftime("%d-%b-%Y %H:%M:%S")
print(f'Test time end: {time_stamp_srt}')

########################################################################################################################
# [5.] Explainable AI ==================================================================================================
########################################################################################################################
print("===============================================================================================================")
print("================ Explainable AI ===============================================================================")
print("===============================================================================================================")
explanation_method = 'saliency'     # 'ig'
output_data_path = os.path.join(os.path.join("data", "output", "KIRC_RANDOM", "plots", "explanations_plots",
                                             # "integrated_gradients"
                                             "saliency"
                                             ))
if os.path.exists(output_data_path):
    shutil.rmtree(output_data_path)
os.mkdir(output_data_path)

for test_idx in range(len(test_dataset)):

    print(f"Test sample index: {test_idx}")

    test_sample_for_explanation = test_dataset[test_idx]

    graph_id_all_array = test_sample_for_explanation.graph_id.split(' ')
    graph_id_array = graph_id_all_array[1].split("_")
    graph_id = graph_id_array[0].replace("id", "")

    node_labels_list = test_sample_for_explanation.node_labels
    ground_truth_label = test_sample_for_explanation.y.cpu().detach().numpy()[0]

    batch_for_prediction = torch.zeros(test_sample_for_explanation.x.shape[0], dtype=int).to(device)
    prediction = model(test_sample_for_explanation.x,
                       test_sample_for_explanation.edge_index,
                       batch_for_prediction).argmax(dim=1).cpu().detach().numpy()[0]

    for explanation_label in [0, 1]:

        print(f"Compute explanation towards label: {explanation_label}")

        edge_mask_relevances = explain_sample(explanation_method, model, test_sample_for_explanation,
                                              explanation_label, device)
        print(f"Min: {min(edge_mask_relevances)}, Max: {max(edge_mask_relevances)}")
        integrated_gradients_viz(test_sample_for_explanation, graph_id, edge_mask_relevances,
                                 node_labels_list, ground_truth_label, prediction, explanation_label,
                                 output_data_path)
    print("===========================================================================================================")

########################################################################################################################
# [6.] Store the GNN ===================================================================================================
########################################################################################################################
# gnn_storage_folder = os.path.join("data", "output", "gnns")
# gnn_model_file_path = os.path.join(gnn_storage_folder, "gcn_model.pth")
# torch.save(model, gnn_model_file_path)

