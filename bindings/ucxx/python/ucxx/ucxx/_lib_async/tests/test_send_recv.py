# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import functools

import pytest

import ucxx
from ucxx._lib_async.utils_test import (
    recv_close_receipt,
    send_close_receipt,
    wait_listener_client_handlers,
)

np = pytest.importorskip("numpy")

msg_sizes = [0, *(2**i for i in range(0, 25, 4))]
dtypes = ["|u1", "<i8", "f8"]


def make_echo_server(create_empty_data):
    """
    Returns an echo server that calls the function `create_empty_data(nbytes)`
    to create the data container.`
    """

    async def echo_server(ep):
        """
        Basic echo server for sized messages.
        We expect the other endpoint to follow the pattern::
        # size of the real message (in bytes)
        >>> await ep.send(msg_size)
        >>> await ep.send(msg)       # send the real message
        >>> await ep.recv(responds)  # receive the echo
        """
        msg_size = np.empty(1, dtype=np.uint64)
        await ep.recv(msg_size)
        msg = create_empty_data(msg_size[0])
        await ep.recv(msg)
        await ep.send(msg)
        await recv_close_receipt(ep)
        await ep.close()

    return echo_server


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
async def test_send_recv_bytes(size):
    msg = bytearray(b"m" * size)
    msg_size = np.array([len(msg)], dtype=np.uint64)

    listener = ucxx.create_listener(make_echo_server(lambda n: bytearray(n)))
    client = await ucxx.create_endpoint(ucxx.get_address(), listener.port)
    await client.send(msg_size)
    await client.send(msg)
    resp = bytearray(size)
    await client.recv(resp)
    assert resp == msg
    await send_close_receipt(client)
    await wait_listener_client_handlers(listener)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("dtype", dtypes)
async def test_send_recv_numpy(size, dtype):
    msg = np.arange(size, dtype=dtype)
    msg_size = np.array([msg.nbytes], dtype=np.uint64)

    listener = ucxx.create_listener(
        make_echo_server(lambda n: np.empty(n, dtype=np.uint8))
    )
    client = await ucxx.create_endpoint(ucxx.get_address(), listener.port)
    await client.send(msg_size)
    await client.send(msg)
    resp = np.empty_like(msg)
    await client.recv(resp)
    np.testing.assert_array_equal(resp, msg)
    await send_close_receipt(client)
    await wait_listener_client_handlers(listener)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.rerun_on_failure(3)
async def test_send_recv_cupy(size, dtype):
    cupy = pytest.importorskip("cupy")

    msg = cupy.arange(size, dtype=dtype)
    msg_size = np.array([msg.nbytes], dtype=np.uint64)

    listener = ucxx.create_listener(
        make_echo_server(lambda n: cupy.empty((n,), dtype=np.uint8))
    )
    client = await ucxx.create_endpoint(ucxx.get_address(), listener.port)
    await client.send(msg_size)
    await client.send(msg)
    resp = cupy.empty_like(msg)
    await client.recv(resp)
    np.testing.assert_array_equal(cupy.asnumpy(resp), cupy.asnumpy(msg))
    await send_close_receipt(client)
    await wait_listener_client_handlers(listener)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.rerun_on_failure(3)
async def test_send_recv_numba(size, dtype):
    cuda = pytest.importorskip("numba.cuda")

    ary = np.arange(size, dtype=dtype)
    msg = cuda.to_device(ary)
    msg_size = np.array([msg.nbytes], dtype=np.uint64)
    listener = ucxx.create_listener(
        make_echo_server(lambda n: cuda.device_array((n,), dtype=np.uint8))
    )
    client = await ucxx.create_endpoint(ucxx.get_address(), listener.port)
    await client.send(msg_size)
    await client.send(msg)
    resp = cuda.device_array_like(msg)
    await client.recv(resp)
    np.testing.assert_array_equal(np.array(resp), np.array(msg))
    await send_close_receipt(client)
    await wait_listener_client_handlers(listener)


@pytest.mark.asyncio
@pytest.mark.skip(reason="See https://github.com/rapidsai/ucxx/issues/104")
async def test_send_recv_error():
    async def say_hey_server(ep):
        await ep.send(bytearray(b"Hey"))
        await ep.close()

    listener = ucxx.create_listener(say_hey_server)
    client = await ucxx.create_endpoint(ucxx.get_address(), listener.port)

    msg = bytearray(100)
    # TODO: remove "Message truncated" match when Python futures accept custom
    # exception messages.
    with pytest.raises(
        ucxx.exceptions.UCXMessageTruncatedError,
        match=r"length mismatch: 3 \(got\) != 100 \(expected\)|Message truncated",
    ):
        await client.recv(msg)
    await wait_listener_client_handlers(listener)
    await client.close()
    listener.close()


@pytest.mark.asyncio
async def test_send_recv_obj():
    async def echo_obj_server(ep):
        obj = await ep.recv_obj()
        await ep.send_obj(obj)
        await recv_close_receipt(ep)

    listener = ucxx.create_listener(echo_obj_server)
    client = await ucxx.create_endpoint(ucxx.get_address(), listener.port)

    msg = bytearray(b"hello")
    await client.send_obj(msg)
    got = await client.recv_obj()
    assert msg == got
    await send_close_receipt(client)
    await wait_listener_client_handlers(listener)


@pytest.mark.asyncio
async def test_send_recv_obj_numpy():
    allocator = functools.partial(np.empty, dtype=np.uint8)

    async def echo_obj_server(ep):
        obj = await ep.recv_obj(allocator=allocator)
        await ep.send_obj(obj)
        await recv_close_receipt(ep)

    listener = ucxx.create_listener(echo_obj_server)
    client = await ucxx.create_endpoint(ucxx.get_address(), listener.port)

    msg = bytearray(b"hello")
    await client.send_obj(msg)
    got = await client.recv_obj(allocator=allocator)
    assert msg == got
    await send_close_receipt(client)
    await wait_listener_client_handlers(listener)
