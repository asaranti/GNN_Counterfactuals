import requests
import string
import random
import jwt
import time
from server.client.utils import send_message

client_host = 'http://localhost:5001'
server = None

class Server:
    def __init__(self):
        jwt_secret = ''.join(random.choices(string.ascii_lowercase, k=64))
        self.status = 'disconnected'
        self.jwt_secret = jwt_secret
    
    def reset_server(self):
        self.status = 'idle'
        del self.server_weights
    
    def send_weights_to_server(self, weights):
        response = self.send_message_to_server('WEIGHTS', weights)
        if not response:
            print('[CLIENT     ] could not send weights to server')
            return
        if response['status'] == 'ok':
            print('[CLIENT     ] sent weights to server')
            self.status = 'waiting_for_average'
            return
        print('[CLIENT     ] received invalid response from server')
    
    def get_weights(self):
        if not self.server_weights: return None
        else: return self.server_weights

    def reveice_weights(self, weights):
        self.server_weights = weights
        print('[CLIENT     ] saved weights from server')
        self.status = 'average_received'
        
    
    def send_message_to_server(self, intent, payload):
        host = self.server_url
        token = self.client_token
        headers = {'Authorization': token, 'X-Clarus-Intent': intent}
        response = send_message(f'{host}/message', payload, headers)
        if not response: 
            print('[CLIENT     ] transmission to server failed')
            return None
        else: 
            print(f'[CLIENT     ] message with intent {intent} transmitted to server')
            #print(response)
            return response
    
    def get_from_server(self, path):
        request_url = self.server_url + path
        token = self.client_token
        headers = {'Authorization': token}
        response = requests.get(request_url, headers=headers)
        return response.json()

    def send_to_server(self, path, payload):
        request_url = self.server_url + path
        token = self.client_token
        headers = {'Authorization': token}
        response = requests.post(request_url, json=payload, headers=headers)
        return response.json()
    
    def get_status(self):
        print('[CLIENT     ] status is ' + self.status)
        return self.status
    
    def disconnect(self):
        print('[CLIENT     ] disconnected from server at ' + self.server_url)
        self.status = 'disconnected'
        self.client_token = None
        self.id = None
    
    def connect(self, url):
        payload = {'id': 'server', 'iat': int(time.time())}
        encoded_jwt = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        token = { 'token': encoded_jwt, 'host': client_host }
        request_url = url + '/register'
        try:
            response = requests.post(request_url, json=token)
            data = response.json()
            self.id = data['id']
            self.server_url = url
            self.client_token = data['token']
            self.status = 'idle'
            print('[CLIENT     ] connected to server at ' + url)
        except:
            print('[CLIENT     ] could not connect to server at ' + url)
            return -1
        return 0
    
    def verify_server_token(self, token):
        try:
            secret = self.jwt_secret
            decoded = jwt.decode(token, secret, algorithms=["HS256"])
            #print(decoded)
            if decoded['id'] == 'server':
                return True
            return False
        except Exception as e:
            #print(e)
            return False


def get_instance():
    global server
    if server == None:
        server = Server()
    return server      