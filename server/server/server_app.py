from flask import Flask, request
import uuid
import time
import random
import string
import jwt
import json
from server_utils import get_client_status, remove_client, start_client_update_loop
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

# Utils
def get_client_by_id(id):
    for client in clients:
        if id == client['id']:
            return client
    return None


if __name__ == "__main__":
    thread = start_client_update_loop(clients, FAIL_THRESHOLD, server_log)
    app.run(debug=True)
