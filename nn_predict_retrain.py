"""
    GNN predict and retrain API

    :author: Anna Saranti
    :copyright: © 2021 HCI-KDD (ex-AI) group
    :date: 2021-11-10
"""

from flask import Flask, request


########################################################################################################################
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
########################################################################################################################
nn_predict_retrain_app = Flask(__name__)


# Start: Index ---------------------------------------------------------------------------------------------------------
@nn_predict_retrain_app.route('/')
@nn_predict_retrain_app.route('/index')
def index():
    return "Hello Predicting and Retraining of the GNN!"


########################################################################################################################
# [1.] Apply the predict() to an already trained GNN ===================================================================
########################################################################################################################
@nn_predict_retrain_app.route('/nn_predict', methods=['GET'])
def nn_predict():
    """
    Apply a new prediction with the current graphs dataset

    :return:
    """


########################################################################################################################
# [2.] Apply the retrain() to an already trained GNN ===================================================================
########################################################################################################################
@nn_predict_retrain_app.route('/nn_retrain', methods=['GET'])
def nn_retrain():
    """
    Apply a new retrain with the current graphs dataset

    :return:
    """


########################################################################################################################
# [3.] Backup ==========================================================================================================
########################################################################################################################
@nn_predict_retrain_app.route('/backup', methods=['GET'])
def backup():
    """
    Backup data and model (snapshot)

    :return:
    """


########################################################################################################################
# MAIN >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
########################################################################################################################
if __name__ == "__main__":
    nn_predict_retrain_app.run(debug=True)
