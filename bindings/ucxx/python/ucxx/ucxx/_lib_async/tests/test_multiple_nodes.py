# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio

import numpy as np
import pytest

import ucxx
from ucxx._lib_async.utils_test import wait_listener_client_handlers


DEFAULT_CONNECT_TIMEOUT = 10.0
MANY_ENDPOINTS_CONNECT_TIMEOUT = 30.0
MANY_ENDPOINTS_THRESHOLD = 50


def get_somaxconn():
    with open("/proc/sys/net/core/somaxconn", "r") as f:
        return int(f.readline())


async def hello(ep):
    msg2send = np.arange(10)
    msg2recv = np.empty_like(msg2send)
    f1 = ep.send(msg2send)
    f2 = ep.recv(msg2recv)
    await f1
    await f2
    np.testing.assert_array_equal(msg2send, msg2recv)
    # assert isinstance(ep.ucx_info(), str)


async def server_node(ep):
    await hello(ep)
    # assert isinstance(ep.ucx_info(), str)


async def client_node(port, connect_timeout):
    ep = await ucxx.create_endpoint(
        ucxx.get_address(), port, connect_timeout=connect_timeout
    )
    await hello(ep)
    await ep.close()
    # assert isinstance(ep.ucx_info(), str)


@pytest.mark.asyncio
@pytest.mark.parametrize("num_servers", [1, 2, 4])
@pytest.mark.parametrize(
    "num_clients",
    [
        1,
        10,
        pytest.param(50, marks=pytest.mark.asyncio_timeout(90)),
        pytest.param(100, marks=pytest.mark.asyncio_timeout(90)),
    ],
)
async def test_many_servers_many_clients(num_servers, num_clients):
    somaxconn = get_somaxconn()
    num_endpoints = num_clients * num_servers
    connect_timeout = (
        MANY_ENDPOINTS_CONNECT_TIMEOUT
        if num_endpoints >= MANY_ENDPOINTS_THRESHOLD
        else DEFAULT_CONNECT_TIMEOUT
    )

    listeners = []

    for _ in range(num_servers):
        listeners.append(
            ucxx.create_listener(server_node, connect_timeout=connect_timeout)
        )

    # We ensure no more than `somaxconn` connections are submitted
    # at once. Doing otherwise can block and hang indefinitely.
    for batch_start in range(0, num_endpoints, somaxconn):
        clients = []
        for endpoint_index in range(
            batch_start, min(batch_start + somaxconn, num_endpoints)
        ):
            clients.append(
                client_node(
                    listeners[endpoint_index % num_servers].port,
                    connect_timeout=connect_timeout,
                )
            )
        await asyncio.gather(*clients)
    await asyncio.gather(
        *(wait_listener_client_handlers(listener) for listener in listeners)
    )
