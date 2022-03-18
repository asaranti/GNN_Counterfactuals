"""
    Flask application instance for the main graph presentation and
    actions on them (addition/removal of nodes and edges as well as features thereof)

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-10-18
"""
import copy
import json
import uuid
import time
import atexit
import os
import re
import numpy as np
import pickle

from flask import Flask, request
from apscheduler.schedulers.background import BackgroundScheduler

from actionable.graph_actions import add_node, add_edge, remove_node, remove_edge, \
    add_feature_all_nodes, remove_feature_all_nodes, add_feature_all_edges, remove_feature_all_edges

from testing_utils.jsonification import graph_to_json

from preprocessing.format_transformations.format_transformation_pytorch_to_ui import transform_from_pytorch_to_ui
from preprocessing.format_transformations.format_transformation_random_kirc_to_pytorch import import_random_kirc_data

from examples.synthetic_graph_examples.ba_graphs_generator import ba_graphs_gen

########################################################################################################################
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
########################################################################################################################
app = Flask(__name__)


# Start: Index ---------------------------------------------------------------------------------------------------------
@app.route('/index')
def index():
    return "Hello Graphs!"


# global
dataset_names = ["Barabasi-Albert Dataset", "Kirc Dataset"]
graph_id_composed_regex = "graph_id_[0-9]+_[0-9]+"
root_folder = os.path.dirname(os.path.abspath(__file__))
# interval to delete old sessions: 5 hours (hour * min * sec * ms)
INTERVAL = 5 * 60 * 60 * 1000
user_last_updated = {}

# Graphs dataset paths -------------------------------------------------------------------------
data_folder = os.path.join(root_folder, "data")


########################################################################################################################
# [0.] Generate Token for current session ==============================================================================
########################################################################################################################
@app.route('/', methods=['GET'])
def initialize():
    token = uuid.uuid4()
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
    if dataset_name == "Barabasi-Albert Dataset":
        # get list of all graphs in pytorch format
        graphs_list = ba_graphs_gen(6, 10, 2, 5, 4)

    elif dataset_name == "Kirc Dataset":
        # get list of all graphs in pytorch format
        dataset_pytorch_folder = os.path.join(data_folder, "output", "KIRC_RANDOM", "kirc_random_pytorch")
        with open(os.path.join(dataset_pytorch_folder, 'kirc_random_nodes_ui_pytorch.pkl'), 'rb') as f:
            graphs_list = pickle.load(f)

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

        # 2.3. Add dict for node_ids -------------------------------------------------------------------------------
        dict_node_ids = {}
        for i in range(0, len(graph.node_ids)):
            dict_node_ids[i] = graph.node_ids[i]
        graph.node_ids = dict_node_ids

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
    node_features = np.array(req_data["features"]).reshape(-1, 1).T

    # Add the node with its features -----------------------------------------------------------------------------------
    output_graph = add_node(input_graph, node_features, node_label, node_id)

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
    deleted_node_id = (list(input_graph.node_ids.keys())[list(input_graph.node_ids.values()).index(deleted_node_id)])

    # delete node
    output_graph = remove_node(input_graph, deleted_node_id, deleted_node_label)

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
    edge_index_left = req_data["new_edge_index_left"]
    edge_index_right = req_data["new_edge_index_right"]
    new_edge_index_left = (list(input_graph.node_ids.keys())[list(input_graph.node_ids.values()).index(edge_index_left)])
    new_edge_index_right = (list(input_graph.node_ids.keys())[list(input_graph.node_ids.values()).index(edge_index_right)])

    # edge features
    try:
        edge_features = np.array(req_data["features"]).reshape(-1, 1).T
    except KeyError:
        edge_features = None

    # Add the node with its features -----------------------------------------------------------------------------------
    output_graph = add_edge(input_graph, new_edge_index_left, new_edge_index_right, edge_features)

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
    edge_index_left = (list(input_graph.node_ids.keys())[list(input_graph.node_ids.values()).index(edge_id_left)])
    edge_index_right = (list(input_graph.node_ids.keys())[list(input_graph.node_ids.values()).index(edge_id_right)])

    # remove edge
    output_graph = remove_edge(input_graph, edge_index_left, edge_index_right)

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
# [9.] Apply the predict() to an already trained GNN ===================================================================
########################################################################################################################
@app.route('/<uuid:token>/nn_predict', methods=['GET'])
def nn_predict(token):
    """
    Apply a new prediction with the current graphs dataset

    :return:

    TODO: The prediction needs to be returned
    """
    # graph and patient id
    patient_id = request.args.get("patient_id")
    graph_id = request.args.get("graph_id")

    # input graph
    graph_data = user_graph_data[str(token)]
    input_graph = graph_data[patient_id][graph_id]

    # create graph in ui format
    nodelist, edgelist = transform_from_pytorch_to_ui(input_graph, "", "", "")

    return json.dumps([nodelist.to_dict(orient='split'), edgelist.to_dict(orient='split')])


########################################################################################################################
# [10.] Apply the retrain() to an already trained GNN ==================================================================
########################################################################################################################
@app.route('/<uuid:token>/nn_retrain', methods=['GET'])
def nn_retrain(token):
    """
    Apply a new retrain with the current graphs dataset

    :return:

    TODO: The retrained graph needs to be returned
    """
    # graph and patient id
    patient_id = request.args.get("patient_id")
    graph_id = request.args.get("graph_id")

    # input graph
    graph_data = user_graph_data[str(token)]
    input_graph = graph_data[patient_id][graph_id]

    # create graph in ui format
    nodelist, edgelist = transform_from_pytorch_to_ui(input_graph, "", "", "")

    return json.dumps([nodelist.to_dict(orient='split'), edgelist.to_dict(orient='split')])


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
    Remove graphs from outdated user sessions (last update > 1 hour)
    """
    # get time in ms
    current_time = round(time.time() * 1000)
    keys_to_remove = []

    if len(user_last_updated) == 0:
        return

    # find outdated session: last modification > 1 hour (INTERVAL)
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
    new_nodes_feature = np.array(req_data["new_nodes_feature"]).reshape(-1, 1)

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
    new_node_feature = np.array(req_data["new_edges_feature"]).reshape(-1, 1)

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
    # scheduler to update / remove sessions (every 5 hours)
    time_in_hours = INTERVAL / 60 / 60 / 1000
    scheduler = BackgroundScheduler(timezone="Europe/Vienna")
    scheduler.add_job(func=remove_session_graphs, trigger="interval", hours=time_in_hours)
    scheduler.start()

    app.run(debug=True)

    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())
