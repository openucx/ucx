# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import multiprocessing
import os
import random

import numpy as np
import pytest

import ucxx
from ucxx._lib_async.utils import get_event_loop
from ucxx._lib_async.utils_test import (
    am_recv,
    am_send,
    compute_timeouts,
    get_cuda_devices,
    get_num_gpus,
    recv,
    send,
    wait_listener_client_handlers,
)
from ucxx.testing import join_processes, run_in_subprocess, terminate_process

cupy = pytest.importorskip("cupy")
rmm = pytest.importorskip("rmm")
distributed = pytest.importorskip("distributed")
cloudpickle = pytest.importorskip("cloudpickle")

# Enable for additional debug output
VERBOSE = False

ITERATIONS = 30


def print_with_pid(msg):
    if VERBOSE:
        print(f"[{os.getpid()}] {msg}")


async def get_ep(name, port):
    addr = ucxx.get_address()
    ep = await ucxx.create_endpoint(addr, port)
    return ep


def client(port, func, comm_api, timeout):
    # 1. Wait for server to come up
    # 2. Loop receiving object multiple times from server
    # 3. Send close message
    # 4. Assert last received message has correct content
    from distributed.utils import nbytes

    # must create context before importing
    # cudf/cupy/etc

    ucxx.init()

    async def read():
        await asyncio.sleep(1)
        ep = await get_ep("client", port)

        for i in range(ITERATIONS):
            print_with_pid(f"Client iteration {i}")
            if comm_api == "tag":
                frames, msg = await recv(ep)
            else:
                while True:
                    try:
                        frames, msg = await am_recv(ep)
                    except ucxx.exceptions.UCXNoMemoryError as e:
                        # Client didn't receive/consume messages quickly enough,
                        # new AM failed to allocate memory and raised this
                        # exception, we need to keep trying.
                        print_with_pid(f"Client exception: {type(e)} {e}")
                    else:
                        break

        close_msg = b"shutdown listener"

        if comm_api == "tag":
            close_msg_size = np.array([len(close_msg)], dtype=np.uint64)

            await ep.send(close_msg_size)
            await ep.send(close_msg)
        else:
            await ep.am_send(close_msg)

        print_with_pid("Shutting Down Client...")
        return msg["data"]

    loop = get_event_loop()
    rx_cuda_obj = loop.run_until_complete(asyncio.wait_for(read(), timeout=timeout))
    rx_cuda_obj + rx_cuda_obj
    num_bytes = nbytes(rx_cuda_obj)
    print_with_pid(f"TOTAL DATA RECEIVED: {num_bytes}")

    cuda_obj_generator = cloudpickle.loads(func)
    pure_cuda_obj = cuda_obj_generator()

    if isinstance(rx_cuda_obj, cupy.ndarray):
        cupy.testing.assert_allclose(rx_cuda_obj, pure_cuda_obj)
    else:
        from cudf.testing import assert_eq

        assert_eq(rx_cuda_obj, pure_cuda_obj)


def server(port, func, comm_api, timeout):
    # 1. Create listener receiver
    # 2. Loop sending object multiple times to connected client
    # 3. Receive close message and close listener
    from distributed.comm.utils import to_frames
    from distributed.protocol import to_serialize

    ucxx.init()

    async def f(listener_port):
        # Coroutine shows up when the client asks to connect
        async def write(ep):
            print_with_pid("CREATING CUDA OBJECT IN SERVER...")
            cuda_obj_generator = cloudpickle.loads(func)
            cuda_obj = cuda_obj_generator()
            msg = {"data": to_serialize(cuda_obj)}
            frames = await to_frames(msg, serializers=("cuda", "dask", "pickle"))
            for i in range(ITERATIONS):
                print_with_pid(f"Server iteration {i}")
                # Send meta data
                if comm_api == "tag":
                    await send(ep, frames)
                else:
                    while True:
                        try:
                            await am_send(ep, frames)
                        except ucxx.exceptions.UCXNoMemoryError as e:
                            # Memory pressure due to client taking too long to
                            # receive will raise an exception.
                            print_with_pid(f"Listener exception: {type(e)} {e}")
                        else:
                            break

            print_with_pid("CONFIRM RECEIPT")
            close_msg = b"shutdown listener"

            if comm_api == "tag":
                msg_size = np.empty(1, dtype=np.uint64)
                await ep.recv(msg_size)

                msg = np.empty(msg_size[0], dtype=np.uint8)
                await ep.recv(msg)
            else:
                msg = await ep.am_recv()

            recv_msg = msg.tobytes()
            assert recv_msg == close_msg
            print_with_pid("Shutting Down Server...")
            await ep.close()
            lf.close()

        lf = ucxx.create_listener(write, port=listener_port)
        await wait_listener_client_handlers(lf)
        try:
            while not lf.closed:
                await asyncio.sleep(0.1)
        except ucxx.UCXCloseError:
            pass

    loop = get_event_loop()
    loop.run_until_complete(asyncio.wait_for(f(port), timeout=timeout))


def dataframe():
    import numpy as np

    import cudf

    # always generate the same random numbers
    np.random.seed(0)
    size = 2**26
    return cudf.DataFrame(
        {"a": np.random.random(size), "b": np.random.random(size)},
        index=np.random.randint(size, size=size),
    )


def series():
    import cudf

    return cudf.Series(np.arange(90000))


def empty_dataframe():
    import cudf

    return cudf.DataFrame({"a": [1.0], "b": [1.0]}).head(0)


def cupy_obj():
    import cupy

    size = 10**8
    return cupy.arange(size)


@pytest.mark.slow
@pytest.mark.skipif(get_num_gpus() <= 2, reason="Machine needs at least two GPUs")
@pytest.mark.parametrize(
    "cuda_obj_generator", [dataframe, empty_dataframe, series, cupy_obj]
)
@pytest.mark.parametrize("comm_api", ["tag", "am"])
def test_send_recv_cu(pytestconfig, cuda_obj_generator, comm_api):
    async_timeout, join_timeout = compute_timeouts(pytestconfig)

    base_env = os.environ
    env_client = base_env.copy()
    # Grab first two devices
    cvd = get_cuda_devices()[:2]
    cvd = ",".join(map(str, cvd))
    # Reverse CVD for client
    env_client["CUDA_VISIBLE_DEVICES"] = cvd[::-1]

    port = random.randint(13000, 15500)

    # Serialize function and send to the client and server. The server will use
    # the return value of the contents, serialize the values, then send
    # serialized values to client. The client will compare return values of the
    # deserialized data sent from the server.
    func = cloudpickle.dumps(cuda_obj_generator)

    ctx = multiprocessing.get_context("spawn")
    server_error_q = ctx.Queue()
    client_error_q = ctx.Queue()
    server_process = ctx.Process(
        name="server",
        target=run_in_subprocess,
        args=(server, server_error_q, port, func, comm_api, async_timeout),
    )
    client_process = ctx.Process(
        name="client",
        target=run_in_subprocess,
        args=(client, client_error_q, port, func, comm_api, async_timeout),
    )

    server_process.start()
    # cuDF will ping the driver for validity of device, this will influence
    # device on which a cuda context is created. Workaround is to update
    # env with new CVD before spawning
    os.environ.update(env_client)
    client_process.start()

    # Increase timeout by an additional 5s to give subprocesses a chance to
    # timeout before being forcefully terminated.
    join_processes([client_process, server_process], timeout=join_timeout)
    terminate_process(client_process, error_queue=client_error_q)
    terminate_process(server_process, error_queue=server_error_q)
