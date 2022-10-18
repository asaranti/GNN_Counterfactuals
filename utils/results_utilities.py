"""
    Results utilities: Transformation methods ...
    :author: Jacqueline Michelle Beinecke
    :copyright: © 2022 HCI-KDD (ex-AI) group
    :date: 2022-07-25
"""

from preprocessing_files.format_transformations.format_transformation_pytorch_to_ui import transform_from_pytorch_to_ui
from actionable.gnn_explanations import explain_sample


def transform_to_results(graph, model, user_token):
    """
    Transform a graph from pytorch format to ui and then append all relevance values for nodes and edges.
    Return in a way that the UI can handle.
    """

    # transform to ui format
    nodelist, edgelist = transform_from_pytorch_to_ui(graph, "", "", "")

    # get node relevances to append to results ---------------------------------------------------------------------
    gnn_exp = list(explain_sample(
        'gnnexplainer',
        model,
        graph.to('cuda:0'),
        int(graph.y.cpu().detach().numpy()[0]),
        user_token,
    ))
    # round value
    gnn_exp = [str(round(node_relevance, 2)) for node_relevance in gnn_exp]

    # append node relevances to nodelist
    nodelist["GNNExplainer"] = gnn_exp

    # get edge relevances to append to results ---------------------------------------------------------------------
    sal = list(explain_sample(
        'saliency',
        model,
        graph.to('cuda:0'),
        int(graph.y.cpu().detach().numpy()[0]),
        user_token,
    ))
    # round value
    sal = [str(round(edge_relevance, 2)) for edge_relevance in sal]
    # append edge relevances to edgelist
    edgelist["Saliency"] = sal

    ig = list(explain_sample(
        'ig',
        model,
        graph.to('cuda:0'),
        int(graph.y.cpu().detach().numpy()[0]),
        user_token,
    ))
    # round value
    ig = [str(round(edge_relevance, 2)) for edge_relevance in ig]

    # append edge relevances to edgelist
    edgelist["IntegratedGradients"] = ig

    return [nodelist.to_dict(orient='split'), edgelist.to_dict(orient='split')]
