# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing as mp
import os

import pytest

import ucxx._lib.libucxx as ucx_api
from ucxx._lib.arr import Array
from ucxx.testing import (
    join_processes,
    run_in_subprocess,
    terminate_process,
    wait_requests,
)

mp = mp.get_context("spawn")


WireupMessageSize = 10


def _close_callback(closed):
    closed[0] = True


def _server(queue, server_close_callback):
    """Server that send received message back to the client

    Notice, since it is illegal to call progress() in call-back functions,
    we use a "chain" of call-back functions.
    """
    ctx = ucx_api.UCXContext(feature_flags=(ucx_api.Feature.TAG,))
    worker = ucx_api.UCXWorker(ctx)

    listener_finished = [False]
    closed = [False]

    # A reference to listener's endpoint is stored to prevent it from going
    # out of scope too early.
    ep = [None]

    def _listener_handler(conn_request):
        ep[0] = listener.create_endpoint_from_conn_request(conn_request, True)
        if server_close_callback is True:
            ep[0].set_close_callback(_close_callback, cb_args=(closed,))
        listener_finished[0] = True

    listener = ucx_api.UCXListener.create(
        worker=worker, port=0, cb_func=_listener_handler
    )
    queue.put(listener.port)

    while ep[0] is None:
        worker.progress()

    wireup_msg_recv = Array(bytearray(WireupMessageSize))
    wireup_msg_send = Array(bytes(os.urandom(WireupMessageSize)))
    wireup_requests = [
        ep[0].tag_recv(wireup_msg_recv, tag=ucx_api.UCXXTag(0)),
        ep[0].tag_send(wireup_msg_send, tag=ucx_api.UCXXTag(0)),
    ]
    wait_requests(worker, "blocking", wireup_requests)

    if server_close_callback is True:
        while closed[0] is False:
            worker.progress()
        assert closed[0] is True
    else:
        while listener_finished[0] is False:
            worker.progress()


def _client(port, server_close_callback):
    ctx = ucx_api.UCXContext(feature_flags=(ucx_api.Feature.TAG,))
    worker = ucx_api.UCXWorker(ctx)
    ep = ucx_api.UCXEndpoint.create(
        worker,
        "127.0.0.1",
        port,
        endpoint_error_handling=True,
    )
    if server_close_callback is False:
        closed = [False]
        ep.set_close_callback(_close_callback, cb_args=(closed,))
    worker.progress()

    wireup_msg_send = Array(bytes(os.urandom(WireupMessageSize)))
    wireup_msg_recv = Array(bytearray(WireupMessageSize))
    wireup_requests = [
        ep.tag_send(wireup_msg_send, tag=ucx_api.UCXXTag(0)),
        ep.tag_recv(wireup_msg_recv, tag=ucx_api.UCXXTag(0)),
    ]
    wait_requests(worker, "blocking", wireup_requests)

    if server_close_callback is False:
        while closed[0] is False:
            worker.progress()


@pytest.mark.parametrize("server_close_callback", [True, False])
def test_close_callback(server_close_callback):
    queue = mp.Queue()
    server_error_q = mp.Queue()
    client_error_q = mp.Queue()
    server = mp.Process(
        target=run_in_subprocess,
        args=(_server, server_error_q, queue, server_close_callback),
    )
    server.start()
    port = queue.get()
    client = mp.Process(
        target=run_in_subprocess,
        args=(_client, client_error_q, port, server_close_callback),
    )
    client.start()
    join_processes([client, server], timeout=10)
    terminate_process(client, error_queue=client_error_q)
    terminate_process(server, error_queue=server_error_q)
