# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import multiprocessing as mp
import os
import struct

import numpy as np
import pytest

import ucxx
from ucxx._lib_async.utils import get_event_loop, hash64bits
from ucxx._lib_async.utils_test import (
    compute_timeouts,
    recv_close_receipt,
    send_close_receipt,
)
from ucxx.testing import join_processes, run_in_subprocess, terminate_process

mp = mp.get_context("spawn")


# Fixed frame size
FRAME_SIZE = 10000

# Header format: Recv Tag (Q) + Send Tag (Q) + UCXAddress.length (Q)
HEADER_STRUCT = struct.Struct("QQQ")


def _test_from_worker_address_server(queue, timeout):
    async def run():
        # Send worker address to client process via multiprocessing.Queue
        address = ucxx.get_worker_address()
        queue.put(address)

        # Receive address size
        address_size = np.empty(1, dtype=np.int64)
        await ucxx.recv(address_size, tag=0)

        # Receive address buffer on tag 0 and create UCXAddress from it
        remote_address = np.empty(address_size[0], dtype=np.uint8)
        await ucxx.recv(remote_address, tag=0)
        remote_address = ucxx.get_ucx_address_from_buffer(remote_address)

        # Create endpoint to remote worker using the received address
        ep = await ucxx.create_endpoint_from_worker_address(remote_address)

        # Send data to client's endpoint
        send_msg = np.arange(10, dtype=np.int64)
        await ep.send(send_msg, tag=1, force_tag=True)
        await recv_close_receipt(ep)
        queue.put("receipt_received")
        await ep.close()

    loop = get_event_loop()
    try:
        loop.run_until_complete(asyncio.wait_for(run(), timeout=timeout))
    finally:
        ucxx.stop_notifier_thread()

        loop.close()


def _test_from_worker_address_client(queue, timeout):
    async def run():
        # Read local worker address
        address = ucxx.get_worker_address()

        # Receive worker address from server via multiprocessing.Queue, create
        # endpoint to server
        remote_address = queue.get()
        ep = await ucxx.create_endpoint_from_worker_address(remote_address)

        # Send local address to server on tag 0
        await ep.send(np.array(address.length, np.int64), tag=0, force_tag=True)
        await ep.send(address, tag=0, force_tag=True)

        # Receive message from server
        recv_msg = np.empty(10, dtype=np.int64)
        await ep.recv(recv_msg, tag=1, force_tag=True)
        await send_close_receipt(ep)
        await asyncio.to_thread(queue.get)
        await ep.close()

        np.testing.assert_array_equal(recv_msg, np.arange(10, dtype=np.int64))

    loop = get_event_loop()
    try:
        loop.run_until_complete(asyncio.wait_for(run(), timeout=timeout))
    finally:
        ucxx.stop_notifier_thread()

        loop.close()


def test_from_worker_address(pytestconfig):
    async_timeout, join_timeout = compute_timeouts(pytestconfig)

    queue = mp.Queue()
    server_error_q = mp.Queue()
    client_error_q = mp.Queue()

    server = mp.Process(
        target=run_in_subprocess,
        args=(_test_from_worker_address_server, server_error_q, queue, async_timeout),
    )
    server.start()

    client = mp.Process(
        target=run_in_subprocess,
        args=(_test_from_worker_address_client, client_error_q, queue, async_timeout),
    )
    client.start()

    join_processes([client, server], timeout=join_timeout)
    terminate_process(client, error_queue=client_error_q)
    terminate_process(server, error_queue=server_error_q)


def _pack_address_and_tag(address, recv_tag, send_tag):
    address_packed = np.empty(FRAME_SIZE, dtype=np.uint8)

    HEADER_STRUCT.pack_into(
        address_packed,  # Buffer to fill
        0,  # Starting Offset
        recv_tag,  # Recv Tag
        send_tag,  # Send Tag
        address.length,  # Address buffer length
    )

    address_start = HEADER_STRUCT.size
    address_stop = address_start + address.length
    address_packed[address_start:address_stop] = memoryview(address)

    return address_packed


def _unpack_address_and_tag(address_packed):
    address_packed = memoryview(address_packed).toreadonly()

    recv_tag, send_tag, address_length = HEADER_STRUCT.unpack_from(
        address_packed,
    )

    address_start = HEADER_STRUCT.size
    address_stop = address_start + address_length
    address = address_packed[address_start:address_stop]

    # Swap send and recv tags, as they are used by the remote process in the
    # opposite direction.
    return {
        "address": address,
        "recv_tag": send_tag,
        "send_tag": recv_tag,
    }


def _test_from_worker_address_server_fixedsize(num_nodes, queue, timeout):
    async def run():
        async def _handle_client(packed_remote_address):
            # Unpack the fixed-size address+tag buffer
            unpacked = _unpack_address_and_tag(packed_remote_address)
            remote_address = ucxx.get_ucx_address_from_buffer(unpacked["address"])

            # Create endpoint to remote worker using the received address
            ep = await ucxx.create_endpoint_from_worker_address(remote_address)

            # Send data to client's endpoint
            send_msg = np.arange(10, dtype=np.int64)
            await ep.send(send_msg, tag=unpacked["send_tag"], force_tag=True)

            # Receive data from client's endpoint
            recv_msg = np.empty(20, dtype=np.int64)
            await ep.recv(recv_msg, tag=unpacked["recv_tag"], force_tag=True)

            np.testing.assert_array_equal(recv_msg, np.arange(20, dtype=np.int64))

        # Send worker address to client processes via multiprocessing.Queue,
        # one entry for each client.
        address = ucxx.get_worker_address()
        for i in range(num_nodes):
            queue.put(address)

        server_tasks = []
        for i in range(num_nodes):
            # Receive fixed-size address+tag buffer on tag 0
            packed_remote_address = np.empty(FRAME_SIZE, dtype=np.uint8)
            await ucxx.recv(packed_remote_address, tag=0)

            # Create an async task for client
            server_tasks.append(_handle_client(packed_remote_address))

        # Await handling each client request
        await asyncio.gather(*server_tasks)

    loop = get_event_loop()
    try:
        loop.run_until_complete(asyncio.wait_for(run(), timeout=timeout))
    finally:
        ucxx.stop_notifier_thread()

        loop.close()


def _test_from_worker_address_client_fixedsize(queue, timeout):
    async def run():
        # Read local worker address
        address = ucxx.get_worker_address()
        recv_tag = hash64bits(os.urandom(16))
        send_tag = hash64bits(os.urandom(16))
        packed_address = _pack_address_and_tag(address, recv_tag, send_tag)

        # Receive worker address from server via multiprocessing.Queue, create
        # endpoint to server
        remote_address = queue.get()
        ep = await ucxx.create_endpoint_from_worker_address(remote_address)

        # Send local address to server on tag 0
        await ep.send(packed_address, tag=0, force_tag=True)

        # Receive message from server
        recv_msg = np.empty(10, dtype=np.int64)
        await ep.recv(recv_msg, tag=recv_tag, force_tag=True)

        np.testing.assert_array_equal(recv_msg, np.arange(10, dtype=np.int64))

        # Send message to server
        send_msg = np.arange(20, dtype=np.int64)
        await ep.send(send_msg, tag=send_tag, force_tag=True)

    loop = get_event_loop()
    try:
        loop.run_until_complete(asyncio.wait_for(run(), timeout=timeout))
    finally:
        ucxx.stop_notifier_thread()

        loop.close()


@pytest.mark.slow
@pytest.mark.xdist_group("test_from_worker_address_multinode")
@pytest.mark.parametrize("num_nodes", [1, 2, 4, 8])
def test_from_worker_address_multinode(pytestconfig, num_nodes):
    async_timeout, join_timeout = compute_timeouts(pytestconfig)

    queue = mp.Queue()
    server_error_q = mp.Queue()

    server = mp.Process(
        target=run_in_subprocess,
        args=(
            _test_from_worker_address_server_fixedsize,
            server_error_q,
            num_nodes,
            queue,
            async_timeout,
        ),
    )
    server.start()

    clients = []
    client_error_qs = []
    for i in range(num_nodes):
        client_error_q = mp.Queue()
        client = mp.Process(
            target=run_in_subprocess,
            args=(
                _test_from_worker_address_client_fixedsize,
                client_error_q,
                queue,
                async_timeout,
            ),
        )
        client.start()
        clients.append(client)
        client_error_qs.append(client_error_q)

    join_processes(clients + [server], timeout=join_timeout)
    for client, client_error_q in zip(clients, client_error_qs):
        terminate_process(client, error_queue=client_error_q)
    terminate_process(server, error_queue=server_error_q)
