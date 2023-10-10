from threading import Thread
import requests
from datetime import datetime
from time import sleep

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