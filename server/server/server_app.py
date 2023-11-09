from flask import Flask, request
import uuid
import time
import random
import string
import jwt
import json
from server_utils import get_client_status, process_weights, remove_client, start_client_update_loop, verify_client_token
from flask_cors import CORS
from server_utils import write_tol_log

FAIL_THRESHOLD = 3

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000'])

jwt_secret = ''.join(random.choices(string.ascii_lowercase, k=64))

clients = []
server_log = []

# Start: Index ---------------------------------------------------------------------------------------------------------
@app.route('/register', methods=['POST'])
def server_register_client():
    req_data = request.get_json()
    server_token = req_data['token']
    client_id = str(uuid.uuid4())
    client_host = req_data['host']
    payload = {'id': client_id, 'iat': int(time.time())}
    encoded_jwt = jwt.encode(payload, jwt_secret, algorithm='HS256')
    token = { 'token': encoded_jwt, 'id': client_id }
    client_data = {
        'server_token': server_token,
        'id': client_id,
        'host': client_host,
        'connection_error': 0,
        'status': 'registered'
    }
    clients.append(client_data)
    write_tol_log('client registered', f'client_{client_id}', server_log)
    return json.dumps(token), 200

@app.route('/log', methods=['GET'])
def server_get_log():
    response = { 'log': server_log }
    return json.dumps(response), 200

@app.route('/clients', methods=['GET'])
def server_get_clients():
    response = { 'clients': [] }
    for client in clients:
        data = {
            'host': client['host'],
            'id': client['id'],
            'status': client['status']
        }
        response['clients'].append(data)
    #print(response)
    return json.dumps(response), 200

@app.route('/message', methods=['POST'])
def receive_server_message():
    headers = request.headers
    client_id = verify_client_token(headers['Authorization'], jwt_secret, clients)
    if not client_id:
        write_tol_log('received unauthorized message', 'server', server_log)
        return json.dumps({ 'error': 'not authorized' }), 401
    intent = None
    try:
        intent = headers['X-Clarus-Intent']
        if not intent: raise "No intent provided"
    except:
        write_tol_log('received message without intent', f'client_{client_id}', server_log)
        return json.dumps({'error': 'no valid intent provided'}), 400
    write_tol_log(f'received message with intent {intent}', f'client_{client_id}', server_log)
    if intent == 'DEBUG':
        write_tol_log(f'message: {request.get_data()}', f'client_{client_id}', server_log)
        return json.dumps({'status': 'ok'}), 200
    if intent == 'WEIGHTS':
        data = request.get_json()
        weights = data['weights']
        client = get_client_by_id(client_id)
        if weights == None:
            del client['weights']
            write_tol_log(f'removed weights of client', f'client_{client_id}', server_log)
        else:
            client['weights'] = weights
            write_tol_log(f'received weights from client', f'client_{client_id}', server_log)
        
        return json.dumps({'status': 'ok'}), 200
    write_tol_log('received message without valid intent', f'client_{client_id}', server_log)
    return json.dumps({'error': 'invalid intent'}), 400

# Utils
def get_client_by_id(id):
    for client in clients:
        if id == client['id']:
            return client
    return None

def check_weights_present(server_log):
    weights_present = 0
    for client in clients:
        if client['weights']: weights_present = weights_present + 1
    write_tol_log(f'{weights_present} of {len(clients)} have submitted weights', f'server', server_log)
    if weights_present == len(clients):
        write_tol_log(f'all clients have submitted weights', f'server', server_log)
        #TODO build average and send to clients
        weights = []
        for client in clients:
            weights.append(client[weights])
        average = process_weights(weights)
    else:
        write_tol_log(f'waiting for weights of {len(clients) - weights_present} clients.', f'server', server_log)


if __name__ == "__main__":
    thread = start_client_update_loop(clients, FAIL_THRESHOLD, server_log)
    app.run(debug=True)
