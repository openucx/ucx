# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import asyncio
import os
from functools import partial

from distributed.comm import connect, listen
from distributed.protocol import Serialized, deserialize, serialize, to_serialize
from distributed.utils_test import gen_test

# Test deserialization
#
# `check_*deserialize` are verbatim copies of Distributed, since they aren't isn't
# exposed publicly can't be expose


async def check_listener_deserialize(addr, deserialize, in_value, check_out):
    q = asyncio.Queue()

    async def handle_comm(comm):
        try:
            msg = await comm.read()
        except Exception as exc:
            q.put_nowait(exc)
        else:
            q.put_nowait(msg)
        finally:
            await comm.close()

    async with listen(addr, handle_comm, deserialize=deserialize) as listener:
        comm = await connect(listener.contact_address)

    await comm.write(in_value)

    out_value = await q.get()
    if isinstance(out_value, Exception):
        raise out_value  # Prevents deadlocks, get actual deserialization exception
    check_out(out_value)
    await comm.close()


async def check_connector_deserialize(addr, deserialize, in_value, check_out):
    done = asyncio.Event()

    async def handle_comm(comm):
        try:
            await comm.write(in_value)
            await done.wait()
        finally:
            await comm.close()

    async with listen(addr, handle_comm) as listener:
        comm = await connect(listener.contact_address, deserialize=deserialize)

    try:
        out_value = await comm.read()
        done.set()
    finally:
        await comm.close()
    check_out(out_value)


async def check_deserialize(addr):
    """
    Check the "deserialize" flag on connect() and listen().
    """
    # Test with Serialize and Serialized objects

    msg = {
        "op": "update",
        "x": b"abc",
        "to_ser": [to_serialize(123)],
        "ser": Serialized(*serialize(456)),
    }
    msg_orig = msg.copy()

    def check_out_false(out_value):
        # Check output with deserialize=False
        out_value = out_value.copy()  # in case transport passed the object as-is
        to_ser = out_value.pop("to_ser")
        ser = out_value.pop("ser")
        expected_msg = msg_orig.copy()
        del expected_msg["ser"]
        del expected_msg["to_ser"]
        assert out_value == expected_msg

        assert isinstance(ser, Serialized)
        assert deserialize(ser.header, ser.frames) == 456

        assert isinstance(to_ser, (tuple, list)) and len(to_ser) == 1
        (to_ser,) = to_ser
        # The to_serialize() value could have been actually serialized
        # or not (it's a transport-specific optimization)
        if isinstance(to_ser, Serialized):
            assert deserialize(to_ser.header, to_ser.frames) == 123
        else:
            assert to_ser == to_serialize(123)

    def check_out_true(out_value):
        # Check output with deserialize=True
        expected_msg = msg.copy()
        expected_msg["ser"] = 456
        expected_msg["to_ser"] = [123]
        # Notice, we allow "to_ser" to be a tuple or a list
        assert list(out_value.pop("to_ser")) == expected_msg.pop("to_ser")
        assert out_value == expected_msg

    await check_listener_deserialize(addr, False, msg, check_out_false)
    await check_connector_deserialize(addr, False, msg, check_out_false)

    await check_listener_deserialize(addr, True, msg, check_out_true)
    await check_connector_deserialize(addr, True, msg, check_out_true)

    # Test with long bytestrings, large enough to be transferred
    # as a separate payload
    # TODO: currently bytestrings are not transferred as a separate payload

    _uncompressible = os.urandom(1024**2) * 4  # end size: 8 MB

    msg = {
        "op": "update",
        "x": _uncompressible,
        "to_ser": (to_serialize(_uncompressible),),
        "ser": Serialized(*serialize(_uncompressible)),
    }
    msg_orig = msg.copy()

    def check_out(deserialize_flag, out_value):
        # Check output with deserialize=False
        assert sorted(out_value) == sorted(msg_orig)
        out_value = out_value.copy()  # in case transport passed the object as-is
        to_ser = out_value.pop("to_ser")
        ser = out_value.pop("ser")
        expected_msg = msg_orig.copy()
        del expected_msg["ser"]
        del expected_msg["to_ser"]
        assert out_value == expected_msg

        if deserialize_flag:
            assert isinstance(ser, (bytes, bytearray))
            assert bytes(ser) == _uncompressible
        else:
            assert isinstance(ser, Serialized)
            assert deserialize(ser.header, ser.frames) == _uncompressible
            assert isinstance(to_ser, tuple) and len(to_ser) == 1
            (to_ser,) = to_ser
            # The to_serialize() value could have been actually serialized
            # or not (it's a transport-specific optimization)
            if isinstance(to_ser, Serialized):
                assert deserialize(to_ser.header, to_ser.frames) == _uncompressible
            else:
                assert to_ser == to_serialize(_uncompressible)

    await check_listener_deserialize(addr, False, msg, partial(check_out, False))
    await check_connector_deserialize(addr, False, msg, partial(check_out, False))

    await check_listener_deserialize(addr, True, msg, partial(check_out, True))
    await check_connector_deserialize(addr, True, msg, partial(check_out, True))


@gen_test()
async def test_ucxx_deserialize(ucxx_loop):
    # Note we see this error on some systems with this test:
    # `socket.gaierror: [Errno -5] No address associated with hostname`
    # This may be due to a system configuration issue.
    await check_deserialize("tcp://")
