# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import asyncio

import numpy as np

import ucxx._lib.libucxx as ucx_api
from ucxx._lib.arr import Array
from ucxx._lib_async.utils import get_event_loop


def _create_cuda_context():
    from ucxx._cuda_context import ensure_cuda_context

    ensure_cuda_context(0)


async def _progress_coroutine(worker):
    while True:
        try:
            if worker is None:
                return
            worker.progress()
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            return


async def _wait_requests_async_future(loop, worker, requests):
    progress_task = loop.create_task(_progress_coroutine(worker))

    await asyncio.gather(*[r.future for r in requests])

    progress_task.cancel()


async def _wait_requests_async_yield(loop, worker, requests):
    progress_task = loop.create_task(_progress_coroutine(worker))

    await asyncio.gather(*[r.wait_yield() for r in requests])

    progress_task.cancel()


def _wait_requests(worker, progress_mode, requests):
    while not all([r.completed for r in requests]):
        if progress_mode == "blocking":
            worker.progress_worker_event()


def parse_args():
    parser = argparse.ArgumentParser(description="Basic UCXX-Py Example")
    parser.add_argument(
        "--asyncio-wait-future",
        default=False,
        action="store_true",
        help="Wait for transfer requests with Python's asyncio using futures "
        "(`UCXRequest.future`), requires `--progress-mode blocking`. "
        "(default: disabled)",
    )
    parser.add_argument(
        "--asyncio-wait-yield",
        default=False,
        action="store_true",
        help="Wait for transfer requests with Python's asyncio by checking "
        "for request completion and yielding (`UCXRequest.wait_yield()`). "
        "(default: disabled)",
    )
    parser.add_argument(
        "-m",
        "--progress-mode",
        default="thread",
        help="Progress mode for the UCP worker. Valid options are: "
        "'thread' (default) and 'blocking'.",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--object-type",
        default="numpy",
        choices=["numpy", "rmm"],
        help="In-memory array type.",
        type=str,
    )
    parser.add_argument(
        "--multi-buffer-transfer",
        default=False,
        action="store_true",
        help="If specified, use the multi-buffer TAG transfer API.",
    )
    parser.add_argument(
        "-p",
        "--port",
        default=12345,
        help="The port the listener will bind to.",
        type=int,
    )

    args = parser.parse_args()
    valid_progress_modes = ["blocking", "thread"]
    if not any(args.progress_mode == v for v in valid_progress_modes):
        raise ValueError(
            f"Unknown progress mode '{args.progress_mode}', "
            f"valid modes are {valid_progress_modes}",
        )
    if args.asyncio_wait_future and args.progress_mode != "blocking":
        raise RuntimeError(
            "`--asyncio-wait-future` requires `--progress-mode='blocking'`"
        )

    return args


def main():
    args = parse_args()

    if args.object_type == "rmm":
        import cupy as xp

        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator

        rmm.reinitialize(
            pool_allocator=True,
            managed_memory=False,
        )
        xp.cuda.runtime.setDevice(0)
        xp.cuda.set_allocator(rmm_cupy_allocator)
    else:
        import numpy as xp

    ctx = ucx_api.UCXContext()

    worker = ucx_api.UCXWorker(ctx)

    if args.progress_mode == "blocking":
        worker.init_blocking_progress_mode()
    else:
        if args.object_type == "rmm":
            worker.set_progress_thread_start_callback(_create_cuda_context)

        worker.start_progress_thread()

    wireup_send_buf = np.arange(3)
    wireup_recv_buf = np.empty_like(wireup_send_buf)
    send_bufs = [
        xp.arange(50, dtype="u1"),
        xp.arange(500, dtype="u1"),
        xp.arange(50000, dtype="u1"),
    ]

    if args.multi_buffer_transfer is False:
        recv_bufs = [np.empty_like(b) for b in send_bufs]

    global listener_ep
    listener_ep = None

    def listener_callback(conn_request):
        global listener_ep
        listener_ep = listener.create_endpoint_from_conn_request(conn_request, True)

    listener = ucx_api.UCXListener.create(
        worker,
        args.port,
        listener_callback,
    )

    ep = ucx_api.UCXEndpoint.create(
        worker,
        "127.0.0.1",
        args.port,
        endpoint_error_handling=True,
    )

    while listener_ep is None:
        if args.progress_mode == "blocking":
            worker.progress_worker_event()

    wireup_requests = [
        ep.tag_send(Array(wireup_send_buf), tag=ucx_api.UCXXTag(0)),
        listener_ep.tag_recv(Array(wireup_recv_buf), tag=ucx_api.UCXXTag(0)),
    ]
    _wait_requests(worker, args.progress_mode, wireup_requests)

    np.testing.assert_equal(wireup_recv_buf, wireup_send_buf)

    if args.multi_buffer_transfer:
        frames = tuple(
            [
                Array(send_bufs[0]),
                Array(send_bufs[1]),
                Array(send_bufs[2]),
            ]
        )

        # data_ptrs = tuple(f.ptr for f in frames)
        # sizes = tuple(f.nbytes for f in frames)
        # is_cuda = tuple(f.cuda for f in frames)
        # send_buffer_requests = listener_ep.tag_send_multi(
        #     data_ptrs, sizes, is_cuda, tag=0
        # )

        send_buffer_requests = listener_ep.tag_send_multi(
            frames, tag=ucx_api.UCXXTag(0)
        )
        recv_buffer_requests = ep.tag_recv_multi(0)

        requests = [send_buffer_requests, recv_buffer_requests]

        if args.asyncio_wait_future:
            loop = get_event_loop()
            loop.run_until_complete(_wait_requests_async_future(loop, worker, requests))
        elif args.asyncio_wait_yield:
            loop = get_event_loop()
            loop.run_until_complete(_wait_requests_async_yield(loop, worker, requests))
        else:
            _wait_requests(worker, args.progress_mode, requests)

            # Check results, raises an exception if any of them failed
            for r in send_buffer_requests.requests + recv_buffer_requests.requests:
                r.check_error()

        recv_bufs = recv_buffer_requests.py_buffers
    else:
        requests = [
            listener_ep.tag_send(Array(send_bufs[0]), tag=ucx_api.UCXXTag(0)),
            listener_ep.tag_send(Array(send_bufs[1]), tag=ucx_api.UCXXTag(1)),
            listener_ep.tag_send(Array(send_bufs[2]), tag=ucx_api.UCXXTag(2)),
            ep.tag_recv(Array(recv_bufs[0]), tag=ucx_api.UCXXTag(0)),
            ep.tag_recv(Array(recv_bufs[1]), tag=ucx_api.UCXXTag(1)),
            ep.tag_recv(Array(recv_bufs[2]), tag=ucx_api.UCXXTag(2)),
        ]

        if args.asyncio_wait_future:
            loop = get_event_loop()
            loop.run_until_complete(_wait_requests_async_future(loop, worker, requests))
        elif args.asyncio_wait_yield:
            loop = get_event_loop()
            loop.run_until_complete(_wait_requests_async_yield(loop, worker, requests))
        else:
            _wait_requests(worker, args.progress_mode, requests)

            # Check results, raises an exception if any of them failed
            for r in requests:
                r.check_error()

    if args.progress_mode == "thread":
        worker.stop_progress_thread()

    for recv_buf, send_buf in zip(recv_bufs, send_bufs):
        if args.object_type == "numpy":
            xp.testing.assert_equal(recv_buf, send_buf)
        else:
            xp.testing.assert_array_equal(xp.asarray(recv_buf), send_buf)


if __name__ == "__main__":
    main()
