# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing as mp
import os
from queue import Empty as QueueIsEmpty

import pytest

import ucxx._lib.libucxx as ucx_api
from ucxx._lib.arr import Array
from ucxx.testing import run_in_subprocess, terminate_process, wait_requests

mp = mp.get_context("spawn")


WireupMessageSize = 10


def _send(ep, api, message):
    if api == "am":
        return ep.am_send(message)
    elif api == "stream":
        return ep.stream_send(message)
    else:
        return ep.tag_send(message, tag=ucx_api.UCXXTag(0))


def _recv(ep, api, message):
    if api == "am":
        return ep.am_recv()
    elif api == "stream":
        return ep.stream_recv(message)
    else:
        return ep.tag_recv(message, tag=ucx_api.UCXXTag(0))


def _echo_server(get_queue, put_queue, transfer_api, msg_size, progress_mode):
    """Server that send received message back to the client

    Notice, since it is illegal to call progress() in call-back functions,
    we keep a reference to the listener's endpoint and execute transfers
    outside of the callback function.
    """
    # TAG is always used for wireup
    feature_flags = [ucx_api.Feature.WAKEUP]
    if transfer_api == "am":
        feature_flags.append(ucx_api.Feature.AM)
    elif transfer_api == "stream":
        feature_flags.append(ucx_api.Feature.STREAM)
    else:
        feature_flags.append(ucx_api.Feature.TAG)

    ctx = ucx_api.UCXContext(feature_flags=tuple(feature_flags))
    worker = ucx_api.UCXWorker(ctx)

    if progress_mode == "blocking":
        worker.init_blocking_progress_mode()
    else:
        worker.start_progress_thread()

    # A reference to listener's endpoint is stored to prevent it from going
    # out of scope too early and allow transfers outside of the listsner's
    # callback even after it has terminated.
    ep = [None]

    def _listener_handler(conn_request):
        ep[0] = listener.create_endpoint_from_conn_request(conn_request, True)

    listener = ucx_api.UCXListener.create(
        worker=worker, port=0, cb_func=_listener_handler
    )
    put_queue.put(listener.port)

    while ep[0] is None:
        if progress_mode == "blocking":
            worker.progress()

    wireup_msg_recv = Array(bytearray(WireupMessageSize))
    wireup_msg_send = Array(bytes(os.urandom(WireupMessageSize)))
    wireup_requests = [
        _recv(ep[0], transfer_api, wireup_msg_recv),
        _send(ep[0], transfer_api, wireup_msg_send),
    ]
    wait_requests(worker, progress_mode, wireup_requests)

    msg = Array(bytearray(msg_size))

    if transfer_api == "stream" and msg_size == 0:
        with pytest.raises(RuntimeError):
            _recv(ep[0], transfer_api, msg)
        with pytest.raises(RuntimeError):
            _send(ep[0], transfer_api, msg)
        return

    # We reuse the message buffer, so we must receive, wait, and then send
    # it back again.
    requests = [_recv(ep[0], transfer_api, msg)]
    wait_requests(worker, progress_mode, requests)
    if transfer_api == "am":
        msg = Array(requests[0].recv_buffer)
    requests = [_send(ep[0], transfer_api, msg)]
    wait_requests(worker, progress_mode, requests)

    while True:
        try:
            get_queue.get(block=True, timeout=0.1)
        except QueueIsEmpty:
            continue
        else:
            break

    if progress_mode == "thread":
        worker.stop_progress_thread()


def _echo_client(transfer_api, msg_size, progress_mode, port):
    # TAG is always used for wireup
    feature_flags = [ucx_api.Feature.WAKEUP]
    if transfer_api == "am":
        feature_flags.append(ucx_api.Feature.AM)
    elif transfer_api == "stream":
        feature_flags.append(ucx_api.Feature.STREAM)
    else:
        feature_flags.append(ucx_api.Feature.TAG)

    ctx = ucx_api.UCXContext(feature_flags=tuple(feature_flags))
    worker = ucx_api.UCXWorker(ctx)

    if progress_mode == "blocking":
        worker.init_blocking_progress_mode()
    else:
        worker.start_progress_thread()

    ep = ucx_api.UCXEndpoint.create(
        worker,
        "127.0.0.1",
        port,
        endpoint_error_handling=True,
    )

    if progress_mode == "blocking":
        worker.progress()

    wireup_msg_send = Array(bytes(os.urandom(WireupMessageSize)))
    wireup_msg_recv = Array(bytearray(WireupMessageSize))
    wireup_requests = [
        _send(ep, transfer_api, wireup_msg_send),
        _recv(ep, transfer_api, wireup_msg_recv),
    ]
    wait_requests(worker, progress_mode, wireup_requests)

    send_msg = bytes(os.urandom(msg_size))
    recv_msg = bytearray(msg_size)

    if transfer_api == "stream" and msg_size == 0:
        with pytest.raises(RuntimeError):
            _send(ep, transfer_api, Array(send_msg))
        with pytest.raises(RuntimeError):
            _recv(ep, transfer_api, Array(recv_msg))
        return

    requests = [
        _send(ep, transfer_api, Array(send_msg)),
        _recv(ep, transfer_api, Array(recv_msg)),
    ]
    wait_requests(worker, progress_mode, requests)

    if transfer_api == "am":
        recv_msg = requests[1].recv_buffer

        assert bytes(recv_msg) == send_msg
    else:
        assert recv_msg == send_msg

    if progress_mode == "thread":
        worker.stop_progress_thread()


@pytest.mark.parametrize("transfer_api", ["am", "stream", "tag"])
@pytest.mark.parametrize("msg_size", [0, 10, 2**24])
@pytest.mark.parametrize("progress_mode", ["blocking", "thread"])
def test_server_client(transfer_api, msg_size, progress_mode):
    put_queue, get_queue = mp.Queue(), mp.Queue()
    server_error_q = mp.Queue()
    client_error_q = mp.Queue()
    server = mp.Process(
        target=run_in_subprocess,
        args=(
            _echo_server,
            server_error_q,
            put_queue,
            get_queue,
            transfer_api,
            msg_size,
            progress_mode,
        ),
    )
    server.start()
    port = get_queue.get()
    client = mp.Process(
        target=run_in_subprocess,
        args=(
            _echo_client,
            client_error_q,
            transfer_api,
            msg_size,
            progress_mode,
            port,
        ),
    )
    client.start()
    client.join(timeout=60)
    terminate_process(client, error_queue=client_error_q)
    put_queue.put("Finished")
    server.join(timeout=10)
    terminate_process(server, error_queue=server_error_q)
