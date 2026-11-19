# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing as mp

import pytest

import ucxx._lib.libucxx as ucx_api
from ucxx._lib.arr import Array
from ucxx.testing import join_processes, run_in_subprocess, terminate_process

mp = mp.get_context("spawn")


def _server_cancel(queue):
    """Server that establishes an endpoint to client and immediately closes
    it, triggering received messages to be canceled on the client.
    """
    ctx = ucx_api.UCXContext(
        feature_flags=(ucx_api.Feature.TAG, ucx_api.Feature.WAKEUP)
    )
    worker = ucx_api.UCXWorker(ctx)

    # Keep endpoint to be used from outside the listener callback
    ep = [None]

    def _listener_handler(conn_request):
        ep[0] = listener.create_endpoint_from_conn_request(conn_request, True)

    listener = ucx_api.UCXListener.create(
        worker=worker, port=0, cb_func=_listener_handler
    )
    queue.put(listener.port)

    while ep[0] is None:
        worker.progress()


def _client_cancel(queue):
    """Client that connects to server and waits for messages to be received,
    because the server closes without sending anything, the messages will
    trigger cancelation.
    """
    ctx = ucx_api.UCXContext(
        feature_flags=(ucx_api.Feature.TAG, ucx_api.Feature.WAKEUP)
    )
    worker = ucx_api.UCXWorker(ctx)
    port = queue.get()
    ep = ucx_api.UCXEndpoint.create(
        worker,
        "127.0.0.1",
        port,
        endpoint_error_handling=True,
    )

    assert ep.alive

    msg = Array(bytearray(1))
    request = ep.tag_recv(msg, tag=ucx_api.UCXXTag(0))

    while not request.completed:
        worker.progress()

    with pytest.raises(ucx_api.UCXCanceledError):
        request.check_error()

    while ep.alive:
        worker.progress()


def test_message_probe():
    queue = mp.Queue()
    server_error_q = mp.Queue()
    client_error_q = mp.Queue()
    server = mp.Process(
        target=run_in_subprocess,
        args=(_server_cancel, server_error_q, queue),
    )
    server.start()
    client = mp.Process(
        target=run_in_subprocess,
        args=(_client_cancel, client_error_q, queue),
    )
    client.start()
    join_processes([client, server], timeout=10)
    terminate_process(client, error_queue=client_error_q)
    terminate_process(server, error_queue=server_error_q)
