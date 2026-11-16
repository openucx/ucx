# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio

from distributed.comm import connect, listen, parse_address
from distributed.comm.registry import backends

import ucxx

from distributed_ucxx.utils_test import gen_test

#
# Test communications through the abstract API
#


async def check_client_server(
    addr,
    check_listen_addr=None,
    check_contact_addr=None,
    listen_args=None,
    connect_args=None,
):
    """
    Abstract client / server test.
    """

    async def handle_comm(comm):
        try:
            scheme, loc = parse_address(comm.peer_address)
            assert scheme == bound_scheme

            msg = await comm.read()
            assert msg["op"] == "ping"
            msg["op"] = "pong"
            await comm.write(msg)

            msg = await comm.read()
            assert msg["op"] == "foobar"
        finally:
            await comm.close()

    # Arbitrary connection args should be ignored
    listen_args = listen_args or {"xxx": "bar"}
    connect_args = connect_args or {"xxx": "foo"}

    listener = await listen(addr, handle_comm, **listen_args)

    # Check listener properties
    bound_addr = listener.listen_address
    bound_scheme, bound_loc = parse_address(bound_addr)
    assert bound_scheme in backends
    assert bound_scheme == parse_address(addr)[0]

    if check_listen_addr is not None:
        check_listen_addr(bound_loc)

    contact_addr = listener.contact_address
    contact_scheme, contact_loc = parse_address(contact_addr)
    assert contact_scheme == bound_scheme

    if check_contact_addr is not None:
        check_contact_addr(contact_loc)
    else:
        assert contact_addr == bound_addr

    # Check client <-> server comms
    keys = []

    async def client_communicate(key, delay=0):
        comm = await connect(listener.contact_address, **connect_args)
        try:
            assert comm.peer_address == listener.contact_address

            await comm.write({"op": "ping", "data": key})
            await comm.write({"op": "foobar"})
            if delay:
                await asyncio.sleep(delay)
            msg = await comm.read()
            assert msg == {"op": "pong", "data": key}
            keys.append(key)
        finally:
            await comm.close()

    await client_communicate(key=1234)

    # Many clients at once
    futures = [client_communicate(key=i, delay=0.05) for i in range(20)]
    await asyncio.gather(*futures)
    assert set(keys) == {1234} | set(range(20))

    listener.stop()


@gen_test()
async def test_ucx_client_server(ucxx_loop):
    addr = ucxx.get_address()
    await check_client_server("ucx://" + addr)
