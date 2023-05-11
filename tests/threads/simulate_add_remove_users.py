import os
from threading import Thread
import pytest
from flask import json

from app import create_app
from app import dataset_names
from app import connected_users
from app import remove_session_graphs
from app import root_folder


# globals
amount_of_users = 1000
list_of_users = []

@pytest.fixture(scope='session')
def app():
    app = create_app()
    app.testing = True
    app.config.root_folder = root_folder
    yield app


@pytest.fixture(scope='session')
def client(app):
    return app.test_client()


def connect_user(client, list_of_users):
    response = client.get('/', content_type='application/json')
    # get tokens
    data = json.loads(response.get_data(as_text=True))
    list_of_users.append(data)


def remove_user():
    remove_session_graphs()


def test_add_users(app, client):
    """
    Test to add 100 users. Dictionary should be thread-safe.
    Expected to pass.
    """
    # configure threads
    threads = list()
    for i in range(amount_of_users):
        thread = Thread(target=connect_user, args=(client, list_of_users))
        threads.append(thread)
    print(f'Created {len(threads)} threads')

    for thread in threads:
        thread.start()
        # wait for threads to finish
    for thread in threads:
        thread.join()

    assert len(list_of_users) == amount_of_users, f"Amount of users in internal list != given amount"
    assert len(connected_users) == amount_of_users, f"Amount of users in global (server) list != given amount"


def test_remove_users_valid(app, client):
    """
    Needs to be run as whole file (session)!
    Test to remove 100 user sessions, which are inactive e.g.
    did not load a dataset.
    Dictionaries should be thread-safe.
    Expected to pass.
    """

    # configure threads
    threads = list()
    for i in range(amount_of_users):
        thread = Thread(target=remove_user)
        threads.append(thread)
    print(f'Created {len(threads)} threads')

    for thread in threads:
        thread.start()
        # wait for threads to finish
    for thread in threads:
        thread.join()

    assert amount_of_users - len(list_of_users) == 0, "Something went wrong! {0} - {1} != 0".format(amount_of_users, len(list_of_users))
    assert len(connected_users) == 0, f"Amount of users in global (server) list != added and deleted amount"
