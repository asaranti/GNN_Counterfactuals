"""
    Parallelization example 2

    :author: Anna Saranti
    :copyright: © 2022 HCI-KDD (ex-AI) group
    :date: 2022-07-22
"""

import copy
import multiprocessing
from multiprocessing import Pool
import os
import pickle
import re
import random
from preprocessing_files.format_transformations.format_transformation_pytorch_to_ui import transform_from_pytorch_to_ui
from preprocessing_files.format_transformations.format_transformation_synth_to_pytorch import import_synthetic_data
from actionable.gnn_explanations import explain_sample


def transform(graph):
    # transform to ui format
    nodelist, edgelist = transform_from_pytorch_to_ui(graph, "", "", "")

    # get node relevances to append to results ---------------------------------------------------------------------
    gnn_exp = list(explain_sample(
        'gnnexplainer',
        graph.to('cpu'),
        int(graph.y.cpu().detach().numpy()[0]),
    ))

    gnn_exp = [str(round(node_relevance, 2)) for node_relevance in gnn_exp]

    # append node relevances to nodelist
    nodelist["GNNExplainer"] = gnn_exp

    # get edge relevances to append to results ---------------------------------------------------------------------
    sal = list(explain_sample(
        'saliency',
        graph.to('cpu'),
        int(graph.y.cpu().detach().numpy()[0]),
    ))

    sal = [str(round(edge_relevance, 2)) for edge_relevance in sal]
    # append edge relevances to edgelist
    edgelist["Saliency"] = sal

    ig = list(explain_sample(
        'ig',
        graph.to('cpu'),
        int(graph.y.cpu().detach().numpy()[0]),
    ))

    ig = [str(round(edge_relevance, 2)) for edge_relevance in ig]
    # append edge relevances to edgelist
    edgelist["IntegratedGradients"] = ig

    pat_results.append([nodelist.to_dict(orient='split'), edgelist.to_dict(orient='split')])

    return pat_results


if __name__ == '__main__':

    ########################################################################################################################
    # MAIN =================================================================================================================
    ########################################################################################################################
    # [1.] Transformation Experiment ::: From Synthetic to Pytorch_Graph ===================================================
    dataset_folder = os.path.join("data", "input", "Synthetic", "synthetic_orig")
    nodes_features_file = "FEATURES_synthetic.txt"
    edges_features_file = "NETWORK_synthetic.txt"
    target_values_features_file = "TARGET_synthetic.txt"

    synthetic_graph_list = import_synthetic_data(
        dataset_folder,
        nodes_features_file,
        edges_features_file,
        target_values_features_file
    )

    # [2.] Select randomly a smaller amount of graphs ======================================================================
    processes_nr = 5

    graphs_nr = 50
    synthetic_graph_50_list = synthetic_graph_list[:graphs_nr]  # random.choices(synthetic_graph_list, k=graphs_nr)

    # [3.] Create dictionary structure ======================================================================
    graph_data = {}
    graph_id_composed_regex = "graph_id_[0-9]+_[0-9]+"
    for graph in synthetic_graph_50_list:
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

    print("Original graph dataset:")
    print(graph_data)

    # [4.] Create list with only lastest graphs (theoretically there could be more graphs per patient) =================
    graph_data_list = []
    for patient_id in range(len(graph_data)):
        # get all modified graphs for this patient
        selected_graphs = graph_data[str(patient_id)]

        # get latest graph id (the indexes start with 0 so subtract length by 1)
        latest_graph_id = len(selected_graphs.keys()) - 1
        latest_graph = graph_data[str(patient_id)][str(latest_graph_id)]

        # get latest modified graph
        graph_data_list.append(latest_graph)

    print(graph_data_list)

    # [5.] Run parallel ================================================================================================
    # the results from transform() should be appended to this list...
    pat_results = []
    print("Parallelize now: ")
    with Pool(processes_nr) as p:
        print(p.map(transform, graph_data_list))
