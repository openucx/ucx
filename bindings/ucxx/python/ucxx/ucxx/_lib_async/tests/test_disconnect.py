# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import logging
import multiprocessing as mp
from io import StringIO
from queue import Empty

import numpy as np
import pytest

import ucxx
from ucxx._lib_async.utils import get_event_loop
from ucxx._lib_async.utils_test import (
    compute_timeouts,
    wait_listener_client_handlers,
)
from ucxx.testing import run_in_subprocess, terminate_process

mp = mp.get_context("spawn")


async def mp_queue_get_nowait(queue):
    while True:
        try:
            return queue.get_nowait()
        except Empty:
            pass
        await asyncio.sleep(0.01)


def _test_shutdown_unexpected_closed_peer_server(
    client_queue, server_queue, endpoint_error_handling, timeout
):
    global ep_alive
    ep_alive = None

    async def run():
        async def server_node(ep):
            try:
                global ep_alive

                await ep.send(np.arange(100, dtype=np.int64))
                # Waiting for signal to close the endpoint
                await mp_queue_get_nowait(server_queue)

                # At this point, the client should have died and the endpoint
                # is not alive anymore. `True` only when endpoint error
                # handling is enabled.
                ep_alive = ep.alive

                await ep.close()
            finally:
                listener.close()

        listener = ucxx.create_listener(
            server_node, endpoint_error_handling=endpoint_error_handling
        )
        client_queue.put(listener.port)
        await wait_listener_client_handlers(listener)
        while not listener.closed:
            await asyncio.sleep(0.1)

    log_stream = StringIO()
    logging.basicConfig(stream=log_stream, level=logging.DEBUG)

    loop = get_event_loop()
    try:
        loop.run_until_complete(asyncio.wait_for(run(), timeout=timeout))
        log = log_stream.getvalue()

        if endpoint_error_handling is True:
            assert ep_alive is False
        else:
            assert ep_alive
            assert log.find("""UCXError('<[Send shutdown]""") != -1
    finally:
        ucxx.stop_notifier_thread()

        loop.close()


def _test_shutdown_unexpected_closed_peer_client(
    client_queue, server_queue, endpoint_error_handling, timeout
):
    async def run():
        server_port = client_queue.get()
        ep = await ucxx.create_endpoint(
            ucxx.get_address(),
            server_port,
            endpoint_error_handling=endpoint_error_handling,
        )
        msg = np.empty(100, dtype=np.int64)
        await ep.recv(msg)

    loop = get_event_loop()
    try:
        loop.run_until_complete(asyncio.wait_for(run(), timeout=timeout))
    finally:
        ucxx.stop_notifier_thread()

        loop.close()


@pytest.mark.parametrize("endpoint_error_handling", [True, False])
def test_shutdown_unexpected_closed_peer(pytestconfig, caplog, endpoint_error_handling):
    """
    Test clean server shutdown after unexpected peer close

    This will causes some UCX warnings to be issued, but this as expected.
    The main goal is to assert that the processes exit without errors
    despite a somewhat messy initial state.
    """
    async_timeout, join_timeout = compute_timeouts(pytestconfig)
    if endpoint_error_handling is False:
        pytest.xfail(
            "Temporarily xfailing, due to https://github.com/rapidsai/ucxx/issues/21"
        )
    if endpoint_error_handling is False and any(
        [
            t.startswith(i)
            for i in ("rc", "dc", "ud")
            for t in ucxx.get_active_transports()
        ]
    ):
        pytest.skip(
            "Endpoint error handling is required when rc, dc or ud transport is enabled"
        )

    client_queue = mp.Queue()
    server_queue = mp.Queue()
    p1_error_q = mp.Queue()
    p2_error_q = mp.Queue()
    p1 = mp.Process(
        target=run_in_subprocess,
        args=(
            _test_shutdown_unexpected_closed_peer_server,
            p1_error_q,
            client_queue,
            server_queue,
            endpoint_error_handling,
            async_timeout,
        ),
    )
    p1.start()
    p2 = mp.Process(
        target=run_in_subprocess,
        args=(
            _test_shutdown_unexpected_closed_peer_client,
            p2_error_q,
            client_queue,
            server_queue,
            endpoint_error_handling,
            async_timeout,
        ),
    )
    p2.start()

    # Increase timeout by an additional 5s to give subprocesses a chance to
    # timeout before being forcefully terminated.
    p2.join(timeout=join_timeout)
    server_queue.put("client is down")
    p1.join(timeout=join_timeout)

    terminate_process(p2, error_queue=p2_error_q)
    terminate_process(p1, error_queue=p1_error_q)
