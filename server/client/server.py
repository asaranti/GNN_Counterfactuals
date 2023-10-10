import requests
import string
import random
import jwt
import time

client_host = 'http://localhost:5001'
server = None

class Server:
    def __init__(self):
        jwt_secret = ''.join(random.choices(string.ascii_lowercase, k=64))
        self.status = 'disconnected'
        self.jwt_secret = jwt_secret
    
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