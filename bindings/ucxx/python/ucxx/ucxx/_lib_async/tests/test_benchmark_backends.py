# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import subprocess
import sys

import pytest

import ucxx
from ucxx._lib.arr import Array
from ucxx._lib_async.utils_test import wait_listener_client_handlers
from ucxx.benchmarks.backends.ucxx_async import _recv_terminal_ack, _send_terminal_ack
from ucxx.benchmarks.send_recv import parse_args

CUDA_UNAVAILABLE_EXIT_CODE = 77


async def _test_async_benchmark_terminal_ack_waits_for_client(enable_am, multi):
    """The server must not close before the client completed its final response."""

    server_received_ack = asyncio.Event()
    server_handler_finished = asyncio.Event()

    async def server_handler(ep):
        if enable_am:
            received = await ep.am_recv()
            await ep.am_send(received)
        elif multi:
            received = await ep.recv_multi()
            await ep.send_multi(received)
        else:
            received = bytearray(1)
            await ep.recv(received)
            await ep.send(received)

        await _recv_terminal_ack(ep, enable_am)
        server_received_ack.set()
        await ep.close()
        server_handler_finished.set()

    listener = ucxx.create_listener(server_handler)
    client = await ucxx.create_endpoint(ucxx.get_address(), listener.port)

    try:
        if enable_am:
            await client.am_send(bytearray(1))
            await client.am_recv()
        elif multi:
            await client.send_multi([bytearray(1), bytearray(1)])
            await client.recv_multi()
        else:
            await client.send(bytearray(1))
            response = bytearray(1)
            await client.recv(response)

        assert not server_received_ack.is_set()
        await _send_terminal_ack(client, enable_am)
        await asyncio.wait_for(server_received_ack.wait(), timeout=5)
        await asyncio.wait_for(server_handler_finished.wait(), timeout=5)
    finally:
        await client.close()
        listener.close()
        await wait_listener_client_handlers(listener)


async def _test_cuda_am_rendezvous_uses_cuda_array_interface():
    import cupy as cp

    message = cp.arange(16 * 1024, dtype=cp.uint8)

    async def server_handler(ep):
        received = await ep.am_recv()
        assert hasattr(received, "__cuda_array_interface__")
        assert Array(received).cuda
        cp.testing.assert_array_equal(cp.asarray(received), message)
        await ep.am_send(received)
        await ep.close()
        listener.close()

    listener = ucxx.create_listener(server_handler)
    client = await ucxx.create_endpoint(ucxx.get_address(), listener.port)

    try:
        await client.am_send(message)
        response = await client.am_recv()
        assert hasattr(response, "__cuda_array_interface__")
        assert Array(response).cuda
        cp.testing.assert_array_equal(cp.asarray(response), message)
    finally:
        await client.close()
        listener.close()
        await wait_listener_client_handlers(listener)


def _cuda_available():
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.parametrize(
    ("enable_am", "multi"), [(False, False), (False, True), (True, False)]
)
def test_async_benchmark_terminal_ack_waits_for_client(enable_am, multi):
    """Run UCX setup outside pytest's process before fork-based tests execute."""

    subprocess.run(
        [sys.executable, __file__, str(enable_am), str(multi)],
        check=True,
        timeout=60,
    )


def test_cuda_am_rendezvous_uses_cuda_array_interface():
    """Run CUDA setup outside pytest before exercising AM rendezvous buffers."""

    result = subprocess.run(
        [sys.executable, __file__, "cuda"],
        check=False,
        timeout=60,
    )
    if result.returncode == CUDA_UNAVAILABLE_EXIT_CODE:
        pytest.skip("CUDA is unavailable")
    result.check_returncode()


if __name__ == "__main__":
    try:
        if sys.argv[1] == "cuda":
            if not _cuda_available():
                sys.exit(CUDA_UNAVAILABLE_EXIT_CODE)
            asyncio.run(_test_cuda_am_rendezvous_uses_cuda_array_interface())
        else:
            asyncio.run(
                _test_async_benchmark_terminal_ack_waits_for_client(
                    sys.argv[1] == "True", sys.argv[2] == "True"
                )
            )
    finally:
        ucxx.reset()


def test_am_benchmark_accepts_device_memory(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["send_recv.py", "--enable-am", "--object_type", "cupy"],
    )

    assert parse_args().enable_am


def test_am_benchmark_accepts_host_memory(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["send_recv.py", "--enable-am"])

    assert parse_args().enable_am
