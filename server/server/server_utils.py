from threading import Thread
import requests
from datetime import datetime
from time import sleep
import jwt

def get_client_status(client):
    #print(client)
    try:
        token = client['server_token']
        host = client['host']
        url = host + '/server/status'
        headers = {'Authorization': token}
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        #print(data)
        return data['status']
    except Exception as e:
        #print(e)
        return 'connection_error'

def write_tol_log(msg, entity, log):
    if len(log) > 999: del log[0]
    now = datetime.now()
    new_entry = {
        'msg': msg,
        'timestamp': now.isoformat(),
        'entity': entity
    }
    log.append(new_entry)
    print(f'[{entity}] {msg}')

def client_update_loop(clients, fail_theshold, server_log):
    while True:
        for client in clients:
            client_status = get_client_status(client)
            client['status'] = client_status
            if client_status == 'connection_error':
                client['connection_error'] = client['connection_error'] + 1
                write_tol_log(f'client connection error, error count: {client["connection_error"]}', f'client_{client["id"]}', server_log)
            if client['connection_error'] > fail_theshold:
                remove_client(client['id'], clients, server_log)
        sleep(5)
        send_message_to_all_clients(clients, "Are you there?", 'DEBUG', server_log)

def remove_client(id, clients, server_log):
    delete_index = -1
    for i in range(len(clients)):
        client = clients[i]
        cid = client['id']
        if id == cid:
            delete_index = i
            break
    if delete_index != -1:
        try:
            del clients[delete_index]
            write_tol_log('client unregistered', f'client_{id}', server_log)
        except Exception as e:
            write_tol_log(f'client deletion failed with {e}', f'client_{id}', server_log)

def start_client_update_loop(clients, fail_theshold, server_log):
    thread = Thread(target = client_update_loop, args = (clients, fail_theshold, server_log))
    thread.start()
    return thread

def send_message(url, payload, headers):
    try:
        response = requests.post(url, payload, headers=headers)
        if response.status_code != 200: raise "Bad Status Code"
    except:
        return 1
    return 0

def send_message_to_client(clients, client_id, payload, intent, server_log):
    res = 0
    for client in clients:
        if client['id'] == client_id:
            host = client['host']
            token = client['server_token']
            headers = {'Authorization': token, 'X-Clarus-Intent': intent}
            response = send_message(f'{host}/server/message', payload, headers)
            if response: 
                write_tol_log(f'message transmission failed', f'client_{client["id"]}', server_log)
                res = 1
            else: 
                write_tol_log(f'message with intent {intent} transmitted', f'client_{client["id"]}', server_log)
                res = 0
    return res

def send_message_to_all_clients(clients, payload, intent, server_log):
    recipients = []
    for client in clients:
        id = client['id']
        response = send_message_to_client(clients, id, payload, intent, server_log)
        if not response: recipients.append(id)
    return recipients

def verify_client_token(token, secret, clients):
    try:
        decoded = jwt.decode(token, secret, algorithms=["HS256"])
        id = decoded['id']
        for client in clients:
            if id == client['id']: return id
        raise 'client not found'
    except Exception as e:
        #print(e)
        return None