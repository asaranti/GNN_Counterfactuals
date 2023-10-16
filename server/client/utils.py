import requests

def send_message(url, payload, headers):
    try:
        response = requests.post(url, payload, headers=headers)
        if response.status_code != 200: raise Exception(f"Bad Status Code: {response.status_code}")
        data = response.json()
        return data
    except Exception as e:
        print(e)
        return None
    return None