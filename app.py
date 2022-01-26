"""
    Flask application instance for the main graph presentation and
    actions on them (addition/removal of nodes and edges as well as features thereof)

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-10-18

    # global
    TODO: Decide on datasets to use an put their names


"""
import copy
import json

from flask import Flask, request

import numpy as np
from torch_geometric.datasets import TUDataset

from actionable.graph_actions import add_node, add_edge, remove_node, remove_edge, \
    add_feature_all_nodes, remove_feature_all_nodes, add_feature_all_edges, remove_feature_all_edges

from testing_utils.jsonification import graph_to_json

from preprocessing.format_transformation_ppi_to_pytorch import transform_from_ppi_to_pytorch
from preprocessing.format_transformation_pytorch_to_ui import transform_from_pytorch_to_ui

from synthetic_graph_examples.ba_graphs_generator import ba_graphs_gen
import os
import re

########################################################################################################################
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
########################################################################################################################
app = Flask(__name__)

# global
dataset_names = ["Barabasi-Albert Dataset"]
root_data_folder = os.getcwd()

# Graphs dataset that was used in the GNN task -------------------------------------------------------------------------
dataset = TUDataset(root='data/TUDataset', name='MUTAG')


########################################################################################################################
# [1.] Add node ========================================================================================================
########################################################################################################################
@app.route('/add_node_json', methods=['POST'])
def adding_node():
    """
    Add a new node and the JSON formatted part of its features
    """

    # Get the features of the node -------------------------------------------------------------------------------------
    req_data = request.get_json()

    # graph and patient id
    patient_id = req_data['patient_id']
    graph_id = req_data['graph_id']
    # input graph
    input_graph = graph_data[str(patient_id)][str(graph_id)]

    # node features
    node_features = np.array(req_data["features"]).reshape(-1, 1).T


    # Add the node with its features -----------------------------------------------------------------------------------
    output_graph = add_node(input_graph, node_features)
    # save graph
    graph_data[str(patient_id)][str(graph_id)] = output_graph

    return "done"


########################################################################################################################
# [2.] Delete node =====================================================================================================
########################################################################################################################
@app.route('/graph_delete_node', methods=['DELETE'])
def delete_node():
    """
    Delete the node from the graph by index

    :param deleted_node_index: Deleted node index
    """

    # graph and patient id
    patient_id = request.args.get('patient_id')
    graph_id = request.args.get('graph_id')

    # get node label
    deleted_node_id = request.args.get('deleted_node_id')

    # input graph
    input_graph = graph_data[str(patient_id)][str(graph_id)]

    # get node id from node label
    deleted_node_id = np.where(input_graph.node_ids == deleted_node_id)

    # delete node
    output_graph = remove_node(input_graph, deleted_node_id)

    # save graph
    graph_data[str(patient_id)][str(graph_id)] = output_graph

    return "done"


########################################################################################################################
# [3.] Add edge ========================================================================================================
########################################################################################################################
@app.route('/add_edge_json', methods=['POST'])
def adding_edge():
    """
    Add a new edge and the JSON formatted part of its features
    """
    # Get the edge's "docking" points and its features -----------------------------------------------------------------
    req_data = request.get_json()

    # graph and patient id
    patient_id = req_data['patient_id']
    graph_id = req_data['graph_id']
    # input graph
    input_graph = graph_data[str(patient_id)][str(graph_id)]

    # left and right node ids
    new_edge_index_left = req_data["new_edge_index_left"]
    new_edge_index_right = req_data["new_edge_index_right"]

    # edge features
    edge_features = np.array(req_data["features"]).reshape(-1, 1).T


    # Add the node with its features -----------------------------------------------------------------------------------
    output_graph = add_edge(input_graph, new_edge_index_left, new_edge_index_right, edge_features)

    # save graph
    graph_data[str(patient_id)][str(graph_id)] = output_graph

    return "done"


########################################################################################################################
# [4.] Delete edge =====================================================================================================
########################################################################################################################
@app.route('/graph_delete_edge', methods=['DELETE'])
def delete_edge():
    """
    Delete the edge from the graph by indexes of the graph nodes that it connects

    :param edge_index_left: Index of left node of the edge
    :param edge_index_right: Index of right node of the edge
    """
    # graph and patient id
    patient_id = request.args.get('patient_id')
    graph_id = request.args.get('graph_id')

    # left and right node id
    edge_id_left = request.args.get('edge_index_left')
    edge_id_right = request.args.get('edge_index_right')
    # input graph
    input_graph = graph_data[str(patient_id)][str(graph_id)]
    # get node ids from node labels
    edge_index_left = np.where(input_graph.node_ids == edge_id_left)
    edge_index_right = np.where(input_graph.node_ids == edge_id_right)

    # remove edge
    output_graph = remove_edge(input_graph, edge_index_left, edge_index_right)

    # save graph
    graph_data[str(patient_id)][str(graph_id)] = output_graph

    return "done"


########################################################################################################################
# [5.] Get all available dataset names =================================================================================
########################################################################################################################
@app.route('/data/dataset_name', methods=['GET'])
def dataset_name():
    """
    Get the dataset_names for the UI
    """
    return json.dumps(dataset_names)

########################################################################################################################
# [6.] Get all available dataset names =================================================================================
########################################################################################################################
@app.route('/data/patient_name', methods=['GET'])
def patient_name():
    """
    Initializes the dataset and gets list of patient names (graph_ids)
    """

    # get dataset_name
    dataset_name = request.args.get('dataset_name')

    # get patient ids corresponding to dataset
    if dataset_name == "Protein Dataset":
        patients_names = ["patient_0"]
    if dataset_name == "Barabasi-Albert Dataset":
        # create the example graphs
        graphs_list = ba_graphs_gen(6, 10, 2, 5, 4)

        # init the structure
        global graph_data
        graph_data = {}
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

            patient_dict = {graph_id: graph}
            graph_data[patient_id] = patient_dict

        # create list of patient names from amount of graphs in dataset
        patients_names = ['Patient ' + i for i in map(str, np.arange(0, len(graph_data)).tolist())]

    return json.dumps(patients_names)


########################################################################################################################
# [7.] Get the actual dataset =========================================================================================
########################################################################################################################
@app.route('/data/dataset/', methods=['GET'])
def pre_defined_dataset():
    """
    Get the dataset information for the UI
    """

    # get dataset_name and patient ID for
    dataset_name = request.args.get('dataset_name')
    patient_id = request.args.get('patient_id')
    graph_id = request.args.get('graph_id')

    #if dataset_name == "Protein Dataset":
    #    dataset_folder = os.path.join(root_data_folder, "data", "Protein_Dataset")
    #    pytorch_ppi_attributes_file = "Human__TCGA_ACC__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct"
    #    pytorch_ppi_node_id_to_name_file = "human.name_2_string.csv"
    #    pytorch_ppi_edges_file = "9606.protein.links.v11.0.txt"

        # perform ppi tp pytorch transformation
    #    pytorch_graph = transform_from_ppi_to_pytorch(dataset_folder,
    #                                 pytorch_ppi_attributes_file,
    #                                 pytorch_ppi_node_id_to_name_file,
    #                                 pytorch_ppi_edges_file)

        # perform pytorch to ui transformation
    #    nodelist, edgelist = transform_from_pytorch_to_ui(pytorch_graph, os.path.join(root_data_folder, "data", "UI_Dataset"),
    #                                                      "nodelist.csv", "edgelist.csv")
    if dataset_name == "Barabasi-Albert Dataset":
        # get graph corresponding to graph id and patient id and transform to UI format
        selected_graph = graph_data[str(patient_id)][str(graph_id)]
        nodelist, edgelist = transform_from_pytorch_to_ui(selected_graph)

    return json.dumps([nodelist.to_dict(orient='split'), edgelist.to_dict(orient='split')])


########################################################################################################################
# [8.] Get all available dataset names =================================================================================
########################################################################################################################
@app.route('/data/performance_values', methods=['GET'])
def performance_values():
    """
    Get the names of all patients

    TODO: Get actual prediction scores
    """

    # get prediction scores
    sens = 31.41
    spec = 27.18
    tp = 30
    tn = 20
    fp = 5
    fn = 0

    return json.dumps([tn, fp, fn, tp, sens, spec])


########################################################################################################################
# [9.] Apply the predict() to an already trained GNN ==================================================================
########################################################################################################################
@app.route('/nn_predict', methods=['GET'])
def nn_predict():
    """
    Apply a new prediction with the current graphs dataset

    :return:

    TODO: The prediction needs to be returned
    """
    # graph and patient id
    patient_id = request.args.get("patient_id")
    graph_id = request.args.get("graph_id")

    # input graph
    input_graph = graph_data[patient_id][graph_id]


    # create graph in ui format
    nodelist, edgelist = transform_from_pytorch_to_ui(input_graph)

    return json.dumps([nodelist.to_dict(orient='split'), edgelist.to_dict(orient='split')])

########################################################################################################################
# [10.] Apply the retrain() to an already trained GNN ==================================================================
########################################################################################################################
@app.route('/nn_retrain', methods=['GET'])
def nn_retrain():
    """
    Apply a new retrain with the current graphs dataset

    :return:

    TODO: The retrained graph needs to be returned
    """
    # graph and patient id
    patient_id = request.args.get("patient_id")
    graph_id = request.args.get("graph_id")

    # input graph
    input_graph = graph_data[patient_id][graph_id]

    # create graph in ui format
    nodelist, edgelist = transform_from_pytorch_to_ui(input_graph)

    return json.dumps([nodelist.to_dict(orient='split'), edgelist.to_dict(orient='split')])


########################################################################################################################
# [11.] Create Deep Copy of graph for modifications ====================================================================
########################################################################################################################
@app.route('/deep_copy', methods=['POST'])
def deep_copy():
    """
    Apply a new retrain with the current graphs dataset

    :return:
    """
    # Get patient id and graph id -----------------------------------------------------------------
    req_data = request.get_json()

    # graph and patient id
    patient_id = req_data["patient_id"]
    graph_id = req_data["graph_id"]

    # input graph
    input_graph = graph_data[str(patient_id)][str(graph_id)]
    print(input_graph)
    # update graph id
    graph_id = int(graph_id) + 1

    # create deep_copy
    deep_cpy = copy.deepcopy(input_graph)

    # update graph id feature
    deep_cpy.graph_id = f"graph_id_{patient_id}_{graph_id}"

    # save graph
    graph_data[str(patient_id)][str(graph_id)] = deep_cpy

    return "done"






### Don't know if needed

########################################################################################################################
# [11.] Backup =========================================================================================================
########################################################################################################################
@app.route('/backup', methods=['GET'])
def backup():
    """
    Backup data and model (snapshot)

    :return:
    """

########################################################################################################################
# [5.] Add feature to all nodes ========================================================================================
########################################################################################################################
@app.route('/add_feature_to_all_nodes_json', methods=['POST'])
def add_feature_to_all_nodes():
    """
    Add a new feature to all nodes.
    Presupposes that the number and interpretation of node features is already known.
    """

    input_graph = dataset[graph_idx]

    # Get the new feature values for all the nodes ---------------------------------------------------------------------
    req_data = request.get_json()
    new_nodes_feature = np.array(req_data["new_nodes_feature"]).reshape(-1, 1)

    # Add the new feature in the graph ---------------------------------------------------------------------------------
    output_graph = add_feature_all_nodes(input_graph, new_nodes_feature)
    output_graph_json = graph_to_json(output_graph)

    return output_graph_json


########################################################################################################################
# [6.] Remove feature to all nodes =====================================================================================
########################################################################################################################
@app.route('/remove_feature_from_all_nodes_json', methods=['DELETE'])
def remove_feature_from_all_nodes():
    """
    Remove a feature from all nodes by index
    """

    input_graph = dataset[graph_idx]

    # Get the features of the node -------------------------------------------------------------------------------------
    req_data = request.get_json()
    removed_node_feature_idx = req_data["removed_nodes_feature_idx"]

    # Remove the feature from all nodes --------------------------------------------------------------------------------
    output_graph = remove_feature_all_nodes(input_graph, removed_node_feature_idx)
    output_graph_json = graph_to_json(output_graph)

    return output_graph_json


########################################################################################################################
# [7.] Add feature to all edges ========================================================================================
########################################################################################################################
@app.route('/add_feature_to_all_edges_json', methods=['POST'])
def add_feature_to_all_edges():
    """
    Add a new feature to all edges.
    Presupposes that the number and interpretation of edge features is already known.
    """

    input_graph = dataset[graph_idx]

    # Get the new feature values for all the nodes ---------------------------------------------------------------------
    req_data = request.get_json()
    new_node_feature = np.array(req_data["new_edges_feature"]).reshape(-1, 1)

    # Add the new feature in the graph ---------------------------------------------------------------------------------
    output_graph = add_feature_all_edges(input_graph, new_node_feature)
    output_graph_json = graph_to_json(output_graph)

    return output_graph_json


########################################################################################################################
# [8.] Remove feature to all edges =====================================================================================
########################################################################################################################
@app.route('/remove_feature_from_all_edges_json', methods=['DELETE'])
def remove_feature_from_all_edges():
    """
    Remove a feature from all edges by index.
    """

    input_graph = dataset[graph_idx]

    # Get the features of the edge -------------------------------------------------------------------------------------
    req_data = request.get_json()
    removed_edge_feature_idx = req_data["removed_edges_feature_idx"]

    # Remove the feature from all edges --------------------------------------------------------------------------------
    output_graph = remove_feature_all_edges(input_graph, removed_edge_feature_idx)
    output_graph_json = graph_to_json(output_graph)

    return output_graph_json


########################################################################################################################
# MAIN >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
########################################################################################################################
if __name__ == "__main__":
    app.run(debug=True)