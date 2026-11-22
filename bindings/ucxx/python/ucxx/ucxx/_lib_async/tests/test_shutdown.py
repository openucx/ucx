# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import logging
import sys

import numpy as np
import pytest

import ucxx as ucxx
from ucxx._lib_async.utils_test import (
    captured_logger,
    wait_listener_client_handlers,
)


async def _shutdown_send(ep, message_type):
    msg = np.arange(10**6)
    if message_type == "tag":
        await ep.send(msg)
    else:
        await ep.am_send(msg)


async def _shutdown_recv(ep, message_type):
    if message_type == "tag":
        msg = np.empty(10**6)
        await ep.recv(msg)
    else:
        await ep.am_recv()


@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", ["tag", "am"])
async def test_server_shutdown(message_type):
    """The server calls shutdown"""

    async def server_node(ep):
        with pytest.raises(ucxx.exceptions.UCXCanceledError):
            await asyncio.gather(_shutdown_recv(ep, message_type), ep.close())
        await ep.close()

    async def client_node(port):
        ep = await ucxx.create_endpoint(
            ucxx.get_address(),
            port,
        )
        with pytest.raises(ucxx.exceptions.UCXCanceledError):
            await _shutdown_recv(ep, message_type)
        await ep.close()

    listener = ucxx.create_listener(
        server_node,
    )
    await client_node(listener.port)
    await wait_listener_client_handlers(listener)
    listener.close()


@pytest.mark.skipif(
    sys.version_info < (3, 7), reason="test currently fails for python3.6"
)
@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", ["tag", "am"])
async def test_client_shutdown(message_type):
    """The client calls shutdown"""

    async def client_node(port):
        ep = await ucxx.create_endpoint(
            ucxx.get_address(),
            port,
        )
        with pytest.raises(ucxx.exceptions.UCXCanceledError):
            await asyncio.gather(_shutdown_recv(ep, message_type), ep.close())
        await ep.close()

    async def server_node(ep):
        with pytest.raises(ucxx.exceptions.UCXCanceledError):
            await _shutdown_recv(ep, message_type)
        await ep.close()

    listener = ucxx.create_listener(
        server_node,
    )
    await client_node(listener.port)
    await wait_listener_client_handlers(listener)
    listener.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", ["tag", "am"])
async def test_listener_close(message_type):
    """The server close the listener"""

    async def client_node(listener):
        ep = await ucxx.create_endpoint(
            ucxx.get_address(),
            listener.port,
        )
        await _shutdown_recv(ep, message_type)
        await _shutdown_recv(ep, message_type)
        assert listener.closed is False
        listener.close()
        assert listener.closed is True

    async def server_node(ep):
        await _shutdown_send(ep, message_type)
        await _shutdown_send(ep, message_type)

    listener = ucxx.create_listener(
        server_node,
    )
    await client_node(listener)
    await wait_listener_client_handlers(listener)


@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", ["tag", "am"])
async def test_listener_del(message_type):
    """The client delete the listener"""

    async def server_node(ep):
        await _shutdown_send(ep, message_type)
        await _shutdown_send(ep, message_type)

    listener = ucxx.create_listener(
        server_node,
    )
    ep = await ucxx.create_endpoint(
        ucxx.get_address(),
        listener.port,
    )
    await _shutdown_recv(ep, message_type)

    assert listener.closed is False
    root = logging.getLogger("ucx")
    with captured_logger(root, level=logging.WARN) as log:
        # Deleting the listener without waiting for all client handlers to complete
        # should be avoided in user code.
        del listener
    assert log.getvalue().startswith("Listener object is being destroyed")

    await _shutdown_recv(ep, message_type)


@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", ["tag", "am"])
async def test_close_after_n_recv(message_type):
    """The Endpoint.close_after_n_recv()"""

    async def server_node(ep):
        for _ in range(10):
            await _shutdown_send(ep, message_type)

    async def client_node(port):
        ep = await ucxx.create_endpoint(
            ucxx.get_address(),
            port,
        )
        ep.close_after_n_recv(10)
        for _ in range(10):
            await _shutdown_recv(ep, message_type)
        assert ep.closed

        ep = await ucxx.create_endpoint(
            ucxx.get_address(),
            port,
        )
        for _ in range(5):
            await _shutdown_recv(ep, message_type)
        ep.close_after_n_recv(5)
        for _ in range(5):
            await _shutdown_recv(ep, message_type)
        assert ep.closed

        ep = await ucxx.create_endpoint(
            ucxx.get_address(),
            port,
        )
        for _ in range(5):
            await _shutdown_recv(ep, message_type)
        ep.close_after_n_recv(10, count_from_ep_creation=True)
        for _ in range(5):
            await _shutdown_recv(ep, message_type)
        assert ep.closed

        ep = await ucxx.create_endpoint(
            ucxx.get_address(),
            port,
        )
        for _ in range(10):
            await _shutdown_recv(ep, message_type)

        with pytest.raises(
            ucxx.exceptions.UCXError,
            match="`n` cannot be less than current recv_count",
        ):
            ep.close_after_n_recv(5, count_from_ep_creation=True)

        ep.close_after_n_recv(1)
        with pytest.raises(
            ucxx.exceptions.UCXError,
            match="close_after_n_recv has already been set to",
        ):
            ep.close_after_n_recv(1)

    listener = ucxx.create_listener(
        server_node,
    )
    await client_node(listener.port)
    await wait_listener_client_handlers(listener)
