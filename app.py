"""
    Flask application instance for the main graph presentation and
    actions on them (addition/removal of nodes and edges as well as features thereof)
    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-10-18
"""
import random

import copy
import json
import uuid
import time
import atexit
import os
import re
import numpy as np
import pickle
import torch
from torch.multiprocessing import Pool
from flask import Flask, request
from apscheduler.schedulers.background import BackgroundScheduler
from functools import partial

from actionable.gnn_actions import GNN_Actions
from actionable.graph_actions import add_node, add_edge, remove_node, remove_edge, \
    add_feature_all_nodes, remove_feature_all_nodes, add_feature_all_edges, remove_feature_all_edges
from actionable.gnn_explanations import explain_sample

from tests.utils_tests.utils_tests_gnns.jsonification import graph_to_json

from preprocessing_files.format_transformations.format_transformation_pytorch_to_ui import transform_from_pytorch_to_ui
from gnns.gnn_selectors.gnn_definitions import define_gnn

from utils.dataset_utilities import keep_only_last_graph_dataset
from utils.results_utilities import transform_to_results
from utils.gnn_load_save import load_gnn_model
########################################################################################################################
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
########################################################################################################################
app = Flask(__name__)

# Start: Index ---------------------------------------------------------------------------------------------------------
@app.route('/index')
def index():
    return "Hello Graphs!"


# global
dataset_names = ["Synthetic Dataset", "KIRC Dataset", "KIRC SubNet"]
graph_id_composed_regex = "graph_id_[0-9]+_[0-9]+"
root_folder = os.path.dirname(os.path.abspath(__file__))
# interval to delete old sessions: 5 hours (hour * min * sec * ms)
INTERVAL = 5 * 60 * 60 * 1000
user_last_updated = {}
connected_users = []
processes_nr = 2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

# Graphs dataset paths -------------------------------------------------------------------------
data_folder = os.path.join(root_folder, "data")



########################################################################################################################
# [0.] Generate Token for current session ==============================================================================
########################################################################################################################
@app.route('/', methods=['GET'])
def initialize():
    token = uuid.uuid4()
    connected_users.append(token)
    return json.dumps(str(token))


########################################################################################################################
# [0.1.] Get all available dataset names ===============================================================================
########################################################################################################################
@app.route('/data/dataset_name', methods=['GET'])
def dataset_name():
    """
    Get the dataset_names for the UI
    """

    return json.dumps(dataset_names)


########################################################################################################################
# [0.2.] Get all available patient names and init data structure =======================================================
########################################################################################################################
@app.route('/<uuid:token>/data/patient_name', methods=['GET'])
def patient_name(token):
    """
    Initializes the dataset and gets list of patient names (graph_ids)
    """
    # get dataset_name
    dataset_name = request.args.get('dataset_name')

    # init the structure
    global user_graph_data
    user_graph_data = {}
    graph_data = {}

    # get patient ids corresponding to dataset
    if dataset_name == "KIRC SubNet":       # get list of all graphs in pytorch format
        dataset_pytorch_folder = os.path.join(data_folder, "output", "KIRC_RANDOM", "kirc_random_pytorch")
        with open(os.path.join(dataset_pytorch_folder, 'kirc_subnet_pytorch.pkl'), 'rb') as f:
            graphs_list = pickle.load(f)
        # load model
        model = load_gnn_model("kirc_subnet", True, str(token))["model"]

    elif dataset_name == "KIRC Dataset":                # get list of all graphs in pytorch format
        dataset_pytorch_folder = os.path.join(data_folder, "output", "KIRC_RANDOM", "kirc_random_pytorch")
        with open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), 'rb') as f:
            graphs_list = pickle.load(f)
        # load model
        model = load_gnn_model("kirc_random_nodes_ui", True, str(token))["model"]


    elif dataset_name == "Synthetic Dataset":           # get list of all graphs in pytorch format
        dataset_pytorch_folder = os.path.join(data_folder, "output", "Synthetic", "synthetic_pytorch")
        with open(os.path.join(dataset_pytorch_folder, 'synthetic_pytorch.pkl'), 'rb') as f:
            graphs_list = pickle.load(f)
        # load model
        model = load_gnn_model("synthetic", True, str(token))["model"]

    # turn list into dictionary format
    for graph in graphs_list:
        # 2.1. Use the graph_id to "position" the graph into the "graph_adaptation_structure" ----------------------
        graph_id_composed = graph.graph_id
        pattern = re.compile(graph_id_composed_regex)
        graph_id_matches = bool(pattern.match(graph_id_composed))

        assert graph_id_matches, f"The graph's id {graph_id_composed} does not match " \
                                 f"the required pattern: {graph_id_composed_regex}"

        # 2.2. Create the initial "graph_adaptation_structure" -----------------------------------------------------
        graph_id_comp_array = graph_id_composed.split("_")
        patient_id = graph_id_comp_array[2]
        graph_id = graph_id_comp_array[3]

        # 2.3. Reformat ndarrays to lists, according to graph_constraints ----------------------------------------------
        graph.node_labels = graph.node_labels
        graph.node_ids = graph.node_ids

        patient_dict = {graph_id: graph}
        graph_data[patient_id] = patient_dict

    # create list of patient names from amount of graphs in dataset
    patients_names = ['Patient ' + i for i in map(str, np.arange(0, len(graph_data)).tolist())]

    # save graph and session id
    user_graph_data[str(token)] = graph_data

    # save user id (token) and last updated time in ms
    user_last_updated[str(token)] = round(time.time() * 1000)

    return json.dumps(patients_names)


########################################################################################################################
# [0.3.] Get the actual dataset ========================================================================================
########################################################################################################################
@app.route('/<uuid:token>/data/dataset/', methods=['GET'])
def pre_defined_dataset(token):
    """
    Get the dataset information for the UI
    """

    # get dataset_name and patient ID for
    dataset_name = request.args.get('dataset_name')
    patient_id = request.args.get('patient_id')
    graph_id = request.args.get('graph_id')

    # get graph corresponding to graph id and patient id and transform to UI format
    graph_data = user_graph_data[str(token)]
    selected_graph = graph_data[str(patient_id)][str(graph_id)]
    nodelist, edgelist = transform_from_pytorch_to_ui(selected_graph, "", "", "")

    return json.dumps([nodelist.to_dict(orient='split'), edgelist.to_dict(orient='split')])


########################################################################################################################
# [1.] Add node ========================================================================================================
########################################################################################################################
@app.route('/<uuid:token>/add_node_json', methods=['POST'])
def adding_node(token):
    """
    Add a new node and the JSON formatted part of its features
    """

    # Get the features of the node -------------------------------------------------------------------------------------
    req_data = request.get_json()
    patient_id = req_data['patient_id']
    graph_id = req_data['graph_id']
    node_label = req_data['label']
    node_id = req_data['id']

    # input graph
    graph_data = user_graph_data[str(token)]
    input_graph = graph_data[str(patient_id)][str(graph_id)]

    # node features
    node_features = np.array(req_data["features"]).astype(np.float32).reshape(-1, 1).T
    #print('got features: ', req_data["features"])
    #print('features: ', node_features)
    #print('f type: ',  node_features.dtype)

    # Add the node with its features -----------------------------------------------------------------------------------
    output_graph = add_node(input_graph, node_features, node_label, node_id)
    #print('out x: ', output_graph.x.dtype)

    # save graph
    graph_data[str(patient_id)][str(graph_id)] = output_graph
    user_graph_data[str(token)] = graph_data
    user_last_updated[str(token)] = round(time.time() * 1000)

    return "done"


########################################################################################################################
# [2.] Delete node =====================================================================================================
########################################################################################################################
@app.route('/<uuid:token>/graph_delete_node', methods=['DELETE'])
def delete_node(token):
    """
    Delete the node from the graph by index
    """

    # graph and patient id from request
    patient_id = request.args.get('patient_id')
    graph_id = request.args.get('graph_id')
    deleted_node_id = request.args.get('deleted_node_id')
    deleted_node_label = request.args.get('deleted_node_label')

    # input graph
    graph_data = user_graph_data[str(token)]
    input_graph = graph_data[str(patient_id)][str(graph_id)]

    # get node id from node ids
    node_index = np.where(np.asarray(input_graph.node_ids) == deleted_node_id)[0][0]

    output_graph = remove_node(input_graph, node_index)

    # save graph
    graph_data[str(patient_id)][str(graph_id)] = output_graph
    user_graph_data[str(token)] = graph_data
    user_last_updated[str(token)] = round(time.time() * 1000)

    return "done"


########################################################################################################################
# [3.] Add edge ========================================================================================================
########################################################################################################################
@app.route('/<uuid:token>/add_edge_json', methods=['POST'])
def adding_edge(token):
    """
    Add a new edge and the JSON formatted part of its features
    """
    # Get the edge's "docking" points and its features -----------------------------------------------------------------
    req_data = request.get_json()
    patient_id = req_data['patient_id']
    graph_id = req_data['graph_id']

    # input graph
    graph_data = user_graph_data[str(token)]
    input_graph = graph_data[str(patient_id)][str(graph_id)]

    # left and right node ids
    edge_id_left = req_data["new_edge_index_left"]
    edge_id_right = req_data["new_edge_index_right"]
    node_index_left = np.where(np.asarray(input_graph.node_ids) == edge_id_left)[0][0]
    node_index_right = np.where(np.asarray(input_graph.node_ids) == edge_id_right)[0][0]

    # edge features
    try:
        edge_features = np.array(req_data["features"]).astype(np.float32).reshape(-1, 1).T
    except KeyError:
        edge_features = None

    # Add the node with its features -----------------------------------------------------------------------------------
    output_graph = add_edge(input_graph, node_index_left, node_index_right, edge_features)

    # save graph
    graph_data[str(patient_id)][str(graph_id)] = output_graph
    user_graph_data[str(token)] = graph_data
    user_last_updated[str(token)] = round(time.time() * 1000)

    return "done"


########################################################################################################################
# [4.] Delete edge =====================================================================================================
########################################################################################################################
@app.route('/<uuid:token>/graph_delete_edge', methods=['DELETE'])
def delete_edge(token):
    """
    Delete the edge from the graph by indexes of the graph nodes that it connects
    """
    # graph and patient id
    patient_id = request.args.get('patient_id')
    graph_id = request.args.get('graph_id')

    # left and right node id
    edge_id_left = request.args.get('edge_index_left')
    edge_id_right = request.args.get('edge_index_right')

    # input graph
    graph_data = user_graph_data[str(token)]
    input_graph = graph_data[str(patient_id)][str(graph_id)]

    # get node ids from node labels
    # for np.ndarrays -> [0][0]
    node_index_left = np.where(np.asarray(input_graph.node_ids) == edge_id_left)[0][0]
    node_index_right = np.where(np.asarray(input_graph.node_ids) == edge_id_right)[0][0]

    # remove edge
    output_graph = remove_edge(input_graph, node_index_left, node_index_right)

    # save graph
    graph_data[str(patient_id)][str(graph_id)] = output_graph
    user_graph_data[str(token)] = graph_data
    user_last_updated[str(token)] = round(time.time() * 1000)

    return "done"


########################################################################################################################
# [8.] Get prediction scores ===========================================================================================
########################################################################################################################
@app.route('/<uuid:token>/data/performance_values', methods=['GET'])
def performance_values(token):
    """
    Get the prediction scores
    """

    return json.dumps(perf_values)


########################################################################################################################
# [9.] Apply the predict() to an already trained GNN ===================================================================
########################################################################################################################
@app.route('/<uuid:token>/nn_predict', methods=['POST'])
def nn_predict(token):
    """
    Apply a new prediction with the current graphs dataset
    :return:
    Prediction of Graph with GNN need to be done
    Return the predicted class
    """

    # Get patient id and graph id --------------------------------------------------------------------------------------
    req_data = request.get_json()
    patient_id = req_data["patient_id"]
    graph_id = req_data["graph_id"]
    dataset_name = req_data["dataset_name"]

    if dataset_name == "KIRC SubNet":       # get list of all graphs in pytorch format
        model = load_gnn_model("kirc_subnet", True, str(token))["model"]
        gnn_architecture_params_dict = define_gnn("kirc_subnet")
        gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, "kirc_subnet")

    elif dataset_name == "KIRC Dataset":                # get list of all graphs in pytorch format
        model = load_gnn_model("kirc_random_nodes_ui", True, str(token))["model"]
        gnn_architecture_params_dict = define_gnn("kirc_random_nodes_ui")
        gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, "kirc_random_nodes_ui")

    elif dataset_name == "Synthetic Dataset":           # get list of all graphs in pytorch format
        model = load_gnn_model("synthetic", True, str(token))["model"]
        gnn_architecture_params_dict = define_gnn("synthetic")
        gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, "synthetic")


    # input graph ------------------------------------------------------------------------------------------------------
    graph_data = user_graph_data[str(token)]
    input_graph = graph_data[str(patient_id)][str(graph_id)]
    input_graph.to(device)

    # predicted class --------------------------------------------------------------------------------------------------
    input_graph.x = input_graph.x.to(dtype=torch.float32)
    predicted_class, prediction_confidence = gnn_actions_obj.gnn_predict(model, input_graph, str(token))

    return "done"


########################################################################################################################
# [10.] Apply the retrain() to an already trained GNN ==================================================================
########################################################################################################################
@app.route('/<uuid:token>/nn_retrain', methods=['POST'])
def nn_retrain(token):
    """
    GNN needs to be retrained on the latest graphs of every patient
    Performance values need to be saved in "perf_values" variable
    :return:
    """
    req_data = request.get_json()
    dataset_name = req_data["dataset_name"]


    # [1.] Get patient id to get the dataset that will be used in retrain ----------------------------------------------
    dataset = user_graph_data[str(token)]

    # [2.] Keep only the last graph in the dataset ---------------------------------------------------------------------
    dataset = keep_only_last_graph_dataset(dataset)

    if dataset_name == "KIRC SubNet":       # get list of all graphs in pytorch format
        model = load_gnn_model("kirc_subnet", True, str(token))["model"]
        gnn_architecture_params_dict = define_gnn("kirc_subnet")
        gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, "kirc_subnet")

    elif dataset_name == "KIRC Dataset":                # get list of all graphs in pytorch format
        model = load_gnn_model("kirc_random_nodes_ui", True, str(token))["model"]
        gnn_architecture_params_dict = define_gnn("kirc_random_nodes_ui")
        gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, "kirc_random_nodes_ui")

    elif dataset_name == "Synthetic Dataset":           # get list of all graphs in pytorch format
        model = load_gnn_model("synthetic", True, str(token))["model"]
        gnn_architecture_params_dict = define_gnn("synthetic")
        gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, "synthetic")

    # [3.] Retrain the GNN ---------------------------------------------------------------------------------------------
    model, test_set_metrics_dict = gnn_actions_obj.gnn_retrain(model, dataset, str(token))

    # [4.] Save performance values in global variable ------------------------------------------------------------------
    global perf_values

    # ----------- [tn, fp, fn, tp, sens, spec] -------------------------------------------------------------------------
    perf_values = [test_set_metrics_dict["true_negatives"],
                   test_set_metrics_dict["false_positives"],
                   test_set_metrics_dict["false_negatives"],
                   test_set_metrics_dict["true_positives"],
                   test_set_metrics_dict["sensitivity"],
                   test_set_metrics_dict["specificity"]]
    return "done"


########################################################################################################################
# [11.] Create Deep Copy of graph for modifications ====================================================================
########################################################################################################################
@app.route('/<uuid:token>/deep_copy', methods=['POST'])
def deep_copy(token):
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
    graph_data = user_graph_data[str(token)]
    input_graph = graph_data[str(patient_id)][str(graph_id)]
    # update graph id
    graph_id = int(graph_id) + 1

    # create deep_copy
    deep_cpy = copy.deepcopy(input_graph)

    # update graph id feature
    deep_cpy.graph_id = f"graph_id_{patient_id}_{graph_id}"

    # save graph
    graph_data[str(patient_id)][str(graph_id)] = deep_cpy
    user_graph_data[str(token)] = graph_data

    return "done"


########################################################################################################################
# [12.] Get highest Graph ID of selected Patient =======================================================================
########################################################################################################################
@app.route('/<uuid:token>/data/highest_graph_id/', methods=['GET'])
def highest_graph_id(token):
    """
    Get highest Graph ID of selected Patient
    """

    # get dataset_name and patient ID for
    patient_id = request.args.get('patient_id')
    # get all graphs of this user session
    graph_data = user_graph_data[str(token)]
    selected_graphs = graph_data[str(patient_id)]
    # count how many graphs (the indexes start with 0 so subtract length by 1)
    amount_graphs = len(selected_graphs.keys())-1

    return json.dumps([amount_graphs])


########################################################################################################################
# [13.] Delete Patient Graph from Dictionary ====================================================================
########################################################################################################################
@app.route('/<uuid:token>/data/graph/', methods=['DELETE'])
def graph(token):
    """
    Remove latest graph of specified patient
    """

    # graph and patient id
    patient_id = request.args.get('patient_id')
    graph_id = request.args.get('graph_id')

    # delete graph
    graph_data = user_graph_data[str(token)]
    del graph_data[str(patient_id)][str(graph_id)]

    return "done"


########################################################################################################################
# [14.] Callback Interval to remove outdated session graphs ============================================================
########################################################################################################################
def remove_session_graphs():
    """
    Remove graphs from outdated user sessions (last update > 5 hour)
    """
    # get time in ms
    current_time = round(time.time() * 1000)
    keys_to_remove = []

    if len(user_last_updated) == 0:
        return

    # find outdated session: last modification > 5 hour (INTERVAL)
    for key, value in user_last_updated.items():
        if value + INTERVAL <= current_time:
            keys_to_remove.append(key)

    # remove all graphs and session of the outdated session
    if len(keys_to_remove) > 0:
        for key in keys_to_remove:
            for graph in user_graph_data[key]:
                del graph
            del user_graph_data[key]
            del user_last_updated[key]

########################################################################################################################
# [15.] Initial training of GNN ========================================================================================
########################################################################################################################
@app.route('/<uuid:token>/gnn', methods=['POST'])
def init_gnn(token):
    """
    Train initial GNN
    Save GNN and performance scores in global variables
    """

    # Get dataset_name -----------------------------------------------------------------
    req_data = request.get_json()

    # graph and patient id
    dataset_name = req_data["dataset_name"]

    # get patient ids corresponding to dataset
    if dataset_name == "KIRC SubNet":  # get list of all graphs in pytorch format
        test_set_metrics_dict = load_gnn_model("kirc_subnet", True, str(token))["test_set_metrics_dict"]

    elif dataset_name == "KIRC Dataset":  # get list of all graphs in pytorch format
        test_set_metrics_dict = load_gnn_model("kirc_random_nodes_ui", True, str(token))["test_set_metrics_dict"]

    elif dataset_name == "Synthetic Dataset":  # get list of all graphs in pytorch format
        test_set_metrics_dict = load_gnn_model("synthetic", True, str(token))["test_set_metrics_dict"]

    # [3.] -------------------------------------------------------------------------------------------------------------
    # save performance values in global variable
    global perf_values

    # ----------- [tn, fp, fn, tp, sens, spec] -------------------------------------------------------------------------
    perf_values = [test_set_metrics_dict["true_negatives"],
                   test_set_metrics_dict["false_positives"],
                   test_set_metrics_dict["false_negatives"],
                   test_set_metrics_dict["true_positives"],
                   test_set_metrics_dict["sensitivity"],
                   test_set_metrics_dict["specificity"]]

    return "done"

########################################################################################################################
# [16.] Get Node importances ===========================================================================================
########################################################################################################################
@app.route('/<uuid:token>/importances/nodes', methods=['GET'])
def node_importance(token):
    """
    Calculate node importance for patient graph
    return: list of importances and corresponding node ids
    TODO: Node importances need to be calculated
    TODO: Node importances need to be returned as as list (see example values)
    """

    # graph and patient id
    patient_id = request.args.get("patient_id")
    graph_id = request.args.get("graph_id")
    method = request.args.get("method")
    dataset_name = request.args.get("dataset_name")

    # current model
    # get patient ids corresponding to dataset
    if dataset_name == "KIRC SubNet":  # get list of all graphs in pytorch format
        model = load_gnn_model("kirc_subnet", False, str(token))["model"]

    elif dataset_name == "KIRC Dataset":  # get list of all graphs in pytorch format
        model = load_gnn_model("kirc_random_nodes_ui", False, str(token))["model"]

    elif dataset_name == "Synthetic Dataset":  # get list of all graphs in pytorch format
        model = load_gnn_model("synthetic", False, str(token))["model"]

    # input graph ------------------------------------------------------------------------------------------------------
    graph_data = user_graph_data[str(token)]
    input_graph = graph_data[patient_id][graph_id]
    input_graph.to(device)

    # get node ids
    node_ids = list(input_graph.node_ids)

    # Explanation ------------------------------------------------------------------------------------------------------
    if method == "gnnexplainer":
        explanation_method = 'gnnexplainer'

    ground_truth_label = int(input_graph.y.cpu().detach().numpy()[0])
    explanation_label = ground_truth_label  # Can also be the opposite - all possible combinations of 0 and 1 ~~~~~~~~~~

    node_mask = explain_sample(explanation_method, model, input_graph, explanation_label, str(token))

    rel_pos = list(explain_sample(
        explanation_method,
        model,
        input_graph,
        explanation_label,
        str(token)
    ))

    rel_pos = [str(round(node_relevance, 2)) for node_relevance in rel_pos]

    # get random positive and negative relevance values
    rel_pos_neg = [random.randint(-100, 100) for p in range(0, len(node_ids))]

    return json.dumps([node_ids, rel_pos, rel_pos_neg])


########################################################################################################################
# [17.] Get Edge importances ===========================================================================================
########################################################################################################################
@app.route('/<uuid:token>/importances/edges', methods=['GET'])
def edge_importance(token):
    """
    Calculate edge importance for patient graph
    return: list of importances and corresponding edge ids
    """

    # graph and patient id ---------------------------------------------------------------------------------------------
    patient_id = request.args.get("patient_id")
    graph_id = request.args.get("graph_id")
    method = request.args.get("method")
    dataset_name = request.args.get("dataset_name")

    # current model
    if dataset_name == "KIRC SubNet":  # get list of all graphs in pytorch format
        model = load_gnn_model("kirc_subnet", False, str(token))["model"]

    elif dataset_name == "KIRC Dataset":  # get list of all graphs in pytorch format
        model = load_gnn_model("kirc_random_nodes_ui", False, str(token))["model"]

    elif dataset_name == "Synthetic Dataset":  # get list of all graphs in pytorch format
        model = load_gnn_model("synthetic", False, str(token))["model"]

    # input graph ------------------------------------------------------------------------------------------------------
    graph_data = user_graph_data[str(token)]
    input_graph = graph_data[patient_id][graph_id]
    input_graph.to(device)

    # get node ids -----------------------------------------------------------------------------------------------------
    edge_ids = list(input_graph.edge_ids)

    # Explanation ------------------------------------------------------------------------------------------------------
    if method == "saliency":
        explanation_method = 'saliency'
    elif method == "ig":
        explanation_method = 'ig'

    ground_truth_label = int(input_graph.y.cpu().detach().numpy()[0])
    explanation_label = ground_truth_label  # Can also be the opposite - all possible combinations of 0 and 1 ~~~~~~~~~~

    edge_mask = explain_sample(explanation_method, model, input_graph, explanation_label, str(token))

    rel_pos = list(explain_sample(
        explanation_method,
        model,
        input_graph,
        explanation_label,
        str(token)
    ))

    rel_pos = [str(round(edge_relevance, 2)) for edge_relevance in rel_pos]

    return json.dumps([edge_ids, rel_pos])


########################################################################################################################
# [18.] Get Patient information ========================================================================================
########################################################################################################################
@app.route('/<uuid:token>/patients/init', methods=['GET'])
def init_patient_information(token):
    """
    Get the following information of the Patient:
    [1.] Is he in Train or Test Data
    [2.] Ground truth label
    [3.] Predicted label
    [4.] Prediction confidence
    """

    # graph and patient id ---------------------------------------------------------------------------------------------
    patient_id = request.args.get("patient_id")
    graph_id = request.args.get("graph_id")
    dataset_name = request.args.get("dataset_name")

    # Ground truth label is already stored -----------------------------------------------------------------------------
    current_graph = user_graph_data[str(token)][patient_id][graph_id]
    ground_truth_label = str(current_graph.y.cpu().detach().numpy()[0])

    # Get its prediction label and prediction performance (or confidence for the prediction) ---------------------------
    # get patient ids corresponding to dataset
    if dataset_name == "KIRC SubNet":  # get list of all graphs in pytorch format
        models_dicts = load_gnn_model("kirc_subnet", True, str(token))

    elif dataset_name == "KIRC Dataset":  # get list of all graphs in pytorch format
        models_dicts = load_gnn_model("kirc_random_nodes_ui", True, str(token))

    elif dataset_name == "Synthetic Dataset":  # get list of all graphs in pytorch format
        models_dicts = load_gnn_model("synthetic", True, str(token))

    # check if patient is in train or test
    if int(patient_id) in list(models_dicts['test_dataset_shuffled_indexes']):
        which_dataset = "Test Data"
    else:
        which_dataset = "Training Data"

    indexes = list(models_dicts['train_dataset_shuffled_indexes']) + list(models_dicts['test_dataset_shuffled_indexes'])
    predicted_labels = models_dicts['train_outputs_predictions_dict'].tolist() + models_dicts['test_outputs_predictions_dict'].tolist()
    prediction_conf = models_dicts['train_prediction_confidence_dict'].tolist() + models_dicts['test_prediction_confidence_dict'].tolist()

    pat_idx = indexes.index(int(patient_id))

    return json.dumps([which_dataset, ground_truth_label, predicted_labels[pat_idx], max(prediction_conf[pat_idx])])

@app.route('/<uuid:token>/patients/pat_info', methods=['GET'])
def patient_information(token):
    """
    Get the following information of the Patient:
    [1.] Is he in Train or Test Data
    [2.] Ground truth label
    [3.] Predicted label
    [4.] Prediction confidence
    """

    # graph and patient id ---------------------------------------------------------------------------------------------
    patient_id = request.args.get("patient_id")
    graph_id = request.args.get("graph_id")
    dataset_name = request.args.get("dataset_name")

    if dataset_name == "KIRC SubNet":       # get list of all graphs in pytorch format
        current_model = load_gnn_model("kirc_subnet", False, str(token))["model"]
        gnn_architecture_params_dict = define_gnn("kirc_subnet")
        gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, "kirc_subnet")

    elif dataset_name == "KIRC Dataset":                # get list of all graphs in pytorch format
        current_model = load_gnn_model("kirc_random_nodes_ui", False, str(token))["model"]
        gnn_architecture_params_dict = define_gnn("kirc_random_nodes_ui")
        gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, "kirc_random_nodes_ui")

    elif dataset_name == "Synthetic Dataset":           # get list of all graphs in pytorch format
        current_model = load_gnn_model("synthetic", False, str(token))["model"]
        gnn_architecture_params_dict = define_gnn("synthetic")
        gnn_actions_obj = GNN_Actions(gnn_architecture_params_dict, "synthetic")

    # Ground truth label is already stored -----------------------------------------------------------------------------
    current_graph = user_graph_data[str(token)][patient_id][graph_id]
    ground_truth_label = str(current_graph.y.cpu().detach().numpy()[0])

    # check if patient is in train or test
    b_is_in_train = gnn_actions_obj.is_in_training_set(current_graph.graph_id)
    which_dataset = "Test Data"
    if b_is_in_train:
        which_dataset = "Training Data"

    predicted_label, prediction_conf = gnn_actions_obj.gnn_predict(current_model, current_graph, str(token))

    return json.dumps([which_dataset, ground_truth_label, predicted_label, prediction_conf])


########################################################################################################################
# [20.] Callback Interval to remove 'outdated' gcn_model files =========================================================
########################################################################################################################
def remove_gcn_model_files():
    """
    Remove gcn_model files from outdated user sessions (modification time > 5 hours)
    """
    # get time in ms
    current_time = round(time.time() * 1000)
    gnn_storage_folder = os.path.join(data_folder, "output", "gnns")

    # find outdated files: last modification > 5 hours (INTERVAL)
    for token in connected_users:
        gcn_model_file_name = "gcn_model_" + str(token) + ".pth"
        gnn_model_file_path = os.path.join(gnn_storage_folder, gcn_model_file_name)
        if os.path.exists(gnn_model_file_path):
            modification_time = os.path.getmtime(gnn_model_file_path)
            if modification_time + INTERVAL <= current_time:
                os.remove(gnn_model_file_path)
                connected_users.remove(token)

########################################################################################################################
# [19.] Save Final Results  ============================================================================================
########################################################################################################################
@app.route('/<uuid:token>/save/results', methods=['GET'])
def results(token):
    # get graph data of user by token
    graph_data = user_graph_data[str(token)]

    # graph and patient id ---------------------------------------------------------------------------------------------
    from_pat = request.args.get("from_pat")
    to_pat = request.args.get("to_pat")
    dataset_name = request.arg.get("dataset_name")

    # [1.] Turn dictionary into a list of graphs =======================================================================
    graph_data_list = []
    for patient_id in range(int(from_pat), int(to_pat)+1):

        # get all modified graphs for this patient
        selected_graphs = graph_data[str(patient_id)]

        # get latest graph id (the indexes start with 0 so subtract length by 1)
        latest_graph_id = len(selected_graphs.keys()) - 1
        latest_graph = graph_data[str(patient_id)][str(latest_graph_id)]

        # get latest modified graph
        graph_data_list.append(latest_graph)

    # current model
    if dataset_name == "KIRC SubNet":  # get list of all graphs in pytorch format
        current_model = load_gnn_model("kirc_subnet", False, str(token))["model"]

    elif dataset_name == "KIRC Dataset":  # get list of all graphs in pytorch format
        current_model = load_gnn_model("kirc_random_nodes_ui", False, str(token))["model"]

    elif dataset_name == "Synthetic Dataset":  # get list of all graphs in pytorch format
        current_model = load_gnn_model("synthetic", False, str(token))["model"]

    # [2.] Run parallel ================================================================================================
    # simplify the method -> user_token stays the same for every execution of a specific user
    transform = partial(transform_to_results, model=current_model, user_token=str(token))

    with Pool(processes_nr) as p:
        pat_results = p.map(transform, graph_data_list)

    return json.dumps(pat_results)

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
@app.route('/<uuid:token>/add_feature_to_all_nodes_json', methods=['POST'])
def add_feature_to_all_nodes(token):
    """
    Add a new feature to all nodes.
    Presupposes that the number and interpretation of node features is already known.
    """

    # Get the new feature values for all the nodes ---------------------------------------------------------------------
    req_data = request.get_json()
    patient_id = req_data["patient_id"]
    graph_id = req_data["graph_id"]
    new_nodes_feature = np.array(req_data["new_nodes_feature"]).astype(np.float32).reshape(-1, 1)

    # get all graphs of this user session
    graph_data = user_graph_data[str(token)]
    input_graph = graph_data[str(patient_id)][str(graph_id)]

    # Add the new feature in the graph ---------------------------------------------------------------------------------
    output_graph = add_feature_all_nodes(input_graph, new_nodes_feature)
    output_graph_json = graph_to_json(output_graph)

    return output_graph_json


########################################################################################################################
# [6.] Remove feature to all nodes =====================================================================================
########################################################################################################################
@app.route('/<uuid:token>/remove_feature_from_all_nodes_json', methods=['DELETE'])
def remove_feature_from_all_nodes(token):
    """
    Remove a feature from all nodes by index
    """

    # Get the features of the node -------------------------------------------------------------------------------------
    req_data = request.get_json()
    patient_id = req_data["patient_id"]
    graph_id = req_data["graph_id"]
    removed_node_feature_idx = req_data["removed_nodes_feature_idx"]

    # get all graphs of this user session
    graph_data = user_graph_data[str(token)]
    input_graph = graph_data[str(patient_id)][str(graph_id)]

    # Remove the feature from all nodes --------------------------------------------------------------------------------
    output_graph = remove_feature_all_nodes(input_graph, removed_node_feature_idx)
    output_graph_json = graph_to_json(output_graph)

    return output_graph_json


########################################################################################################################
# [7.] Add feature to all edges ========================================================================================
########################################################################################################################
@app.route('/<uuid:token>/add_feature_to_all_edges_json', methods=['POST'])
def add_feature_to_all_edges(token):
    """
    Add a new feature to all edges.
    Presupposes that the number and interpretation of edge features is already known.
    """

    # Get the new feature values for all the nodes ---------------------------------------------------------------------
    req_data = request.get_json()
    patient_id = req_data["patient_id"]
    graph_id = req_data["graph_id"]
    new_node_feature = np.array(req_data["new_edges_feature"]).astype(np.float32).reshape(-1, 1)

    # get all graphs of this user session
    graph_data = user_graph_data[str(token)]
    input_graph = graph_data[str(patient_id)][str(graph_id)]

    # Add the new feature in the graph ---------------------------------------------------------------------------------
    output_graph = add_feature_all_edges(input_graph, new_node_feature)
    output_graph_json = graph_to_json(output_graph)

    return output_graph_json


########################################################################################################################
# [8.] Remove feature to all edges =====================================================================================
########################################################################################################################
@app.route('/<uuid:token>/remove_feature_from_all_edges_json', methods=['DELETE'])
def remove_feature_from_all_edges(token):
    """
    Remove a feature from all edges by index.
    """

    # Get the features of the edge -------------------------------------------------------------------------------------
    req_data = request.get_json()
    patient_id = req_data["patient_id"]
    graph_id = req_data["graph_id"]
    removed_edge_feature_idx = req_data["removed_edges_feature_idx"]

    # get all graphs of this user session
    graph_data = user_graph_data[str(token)]
    input_graph = graph_data[str(patient_id)][str(graph_id)]

    # Remove the feature from all edges --------------------------------------------------------------------------------
    output_graph = remove_feature_all_edges(input_graph, removed_edge_feature_idx)
    output_graph_json = graph_to_json(output_graph)

    return output_graph_json


########################################################################################################################
# MAIN >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
########################################################################################################################
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    # scheduler to update / remove sessions (every 5 hours)
    time_in_hours = INTERVAL / 60 / 60 / 1000
    scheduler = BackgroundScheduler(timezone="Europe/Vienna")
    scheduler.add_job(func=remove_session_graphs, trigger="interval", hours=time_in_hours)
    scheduler.add_job(func=remove_gcn_model_files, trigger="interval", hours=time_in_hours)
    scheduler.start()

    app.run(debug=True)

    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())
