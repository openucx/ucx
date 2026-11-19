# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
from queue import Queue
from time import monotonic, sleep

import ucxx
import ucxx._lib.libucxx as ucx_api
from ucxx._lib.arr import Array
from ucxx._lib_async.utils import get_event_loop
from ucxx.benchmarks.backends.base import BaseClient, BaseServer
from ucxx.benchmarks.utils import get_allocator
from ucxx.utils import print_key_value

WireupMessage = bytearray(b"wireup")
TerminalAckTag = ucx_api.UCXXTag(2)


def _create_cuda_context(device):
    from ucxx._cuda_context import ensure_cuda_context

    ensure_cuda_context(device)


def _transfer_wireup(ep, server):
    import numpy as np

    # Using bytearray currently segfaults
    # TODO: fix
    # message = bytearray(b"wireup")

    message = np.array([1], dtype="u8")
    if server:
        message = Array(message)
        return [
            ep.tag_recv(message, tag=ucx_api.UCXXTag(1)),
            ep.tag_send(message, tag=ucx_api.UCXXTag(0)),
        ]
    else:
        message = Array(np.zeros_like(message))
        return [
            ep.tag_send(message, tag=ucx_api.UCXXTag(1)),
            ep.tag_recv(message, tag=ucx_api.UCXXTag(0)),
        ]


async def _wait_requests_async(worker, requests):
    import asyncio

    await asyncio.gather(*[r.wait_yield() for r in requests])


async def _wait_requests_checked(worker, progress_mode, asyncio_wait, requests):
    if asyncio_wait:
        await _wait_requests_async(worker, requests)
    else:
        _wait_requests(worker, progress_mode, requests)
        for request in requests:
            request.check_error()


async def _send_terminal_ack(ep, worker, args):
    """Confirm that the peer completed the final benchmark response."""
    import numpy as np

    ack = Array(np.zeros(1, dtype="u1"))
    request = (
        ep.am_send(ack) if args.enable_am else ep.tag_send(ack, tag=TerminalAckTag)
    )
    await _wait_requests_checked(
        worker, args.progress_mode, args.asyncio_wait, [request]
    )


async def _recv_terminal_ack(ep, worker, args):
    """Wait for the peer to confirm the final benchmark response."""
    import numpy as np

    if args.enable_am:
        request = ep.am_recv()
        await _wait_requests_checked(
            worker, args.progress_mode, args.asyncio_wait, [request]
        )
        ack = request.recv_buffer
    else:
        ack = Array(np.empty(1, dtype="u1"))
        request = ep.tag_recv(ack, tag=TerminalAckTag)
        await _wait_requests_checked(
            worker, args.progress_mode, args.asyncio_wait, [request]
        )
    if ack.nbytes != 1:
        raise RuntimeError("Invalid benchmark terminal acknowledgement")


def _wait_requests(worker, progress_mode, requests):
    while not all([r.completed for r in requests]):
        if progress_mode == "blocking":
            worker.progress_worker_event()
        if progress_mode == "polling":
            worker.progress()


class UCXPyCoreServer(BaseServer):
    has_cuda_support = True

    def __init__(
        self,
        args: Namespace,
        queue: Queue,
    ):
        self.args = args
        self.queue = queue

    def run(self):
        self.ep = None

        ctx = ucx_api.UCXContext(
            feature_flags=(
                ucx_api.Feature.AM if self.args.enable_am else ucx_api.Feature.TAG,
                ucx_api.Feature.WAKEUP,
            )
        )
        worker = ucx_api.UCXWorker(ctx)

        xp = get_allocator(
            self.args.object_type,
            self.args.rmm_init_pool_size,
            self.args.rmm_managed_memory,
        )

        if self.args.progress_mode.startswith("thread"):
            worker.set_progress_thread_start_callback(
                _create_cuda_context, cb_args=(self.args.server_dev,)
            )
            polling_mode = self.args.progress_mode == "thread-polling"
            worker.start_progress_thread(polling_mode=polling_mode)
        else:
            worker.init_blocking_progress_mode()

        # A reference to listener's endpoint is stored to prevent it from going
        # out of scope immediately after the listener callback terminates.
        global ep
        ep = None

        def _listener_handler(conn_request):
            global ep
            ep = listener.create_endpoint_from_conn_request(
                conn_request, endpoint_error_handling=self.args.error_handling
            )

        listener = ucx_api.UCXListener.create(
            worker=worker, port=self.args.port or 0, cb_func=_listener_handler
        )
        self.queue.put(listener.port)

        # Without this, q.get() in main() may sometimes hang indefinitely.
        # TODO: find root cause and fix.
        sleep(0.1)

        while ep is None:
            if self.args.progress_mode == "blocking":
                worker.progress_worker_event()
            elif self.args.progress_mode == "polling":
                worker.progress()

        # Wireup before starting to transfer data
        wireup_requests = _transfer_wireup(ep, server=True)
        _wait_requests(worker, self.args.progress_mode, wireup_requests)

        async def _transfer():
            if self.args.reuse_alloc:
                recv_msg = Array(xp.zeros(self.args.n_bytes, dtype="u1"))

            for i in range(self.args.n_iter + self.args.n_warmup_iter):
                if not self.args.reuse_alloc:
                    recv_msg = Array(xp.zeros(self.args.n_bytes, dtype="u1"))

                if self.args.enable_am:
                    recv_request = ep.am_recv()
                    await _wait_requests_checked(
                        worker,
                        self.args.progress_mode,
                        self.args.asyncio_wait,
                        [recv_request],
                    )
                    send_request = ep.am_send(Array(recv_request.recv_buffer))
                    await _wait_requests_checked(
                        worker,
                        self.args.progress_mode,
                        self.args.asyncio_wait,
                        [send_request],
                    )
                else:
                    requests = [
                        ep.tag_recv(recv_msg, tag=ucx_api.UCXXTag(1)),
                        ep.tag_send(recv_msg, tag=ucx_api.UCXXTag(0)),
                    ]
                    await _wait_requests_checked(
                        worker,
                        self.args.progress_mode,
                        self.args.asyncio_wait,
                        requests,
                    )

            await _recv_terminal_ack(ep, worker, self.args)

        loop = get_event_loop()
        loop.run_until_complete(_transfer())


class UCXPyCoreClient(BaseClient):
    has_cuda_support = True

    def __init__(
        self,
        args: Namespace,
        queue: Queue,
        server_address: str,
        port: int,
    ):
        self.args = args
        self.queue = queue
        self.server_address = server_address
        self.port = port

    def run(self):
        ctx = ucx_api.UCXContext(
            feature_flags=(
                ucx_api.Feature.AM
                if self.args.enable_am is True
                else ucx_api.Feature.TAG,
                ucx_api.Feature.WAKEUP,
            )
        )
        worker = ucx_api.UCXWorker(ctx)

        xp = get_allocator(
            self.args.object_type,
            self.args.rmm_init_pool_size,
            self.args.rmm_managed_memory,
        )
        send_msg = Array(xp.arange(self.args.n_bytes, dtype="u1"))

        if self.args.progress_mode.startswith("thread"):
            worker.set_progress_thread_start_callback(
                _create_cuda_context, cb_args=(self.args.client_dev,)
            )
            polling_mode = self.args.progress_mode == "thread-polling"
            worker.start_progress_thread(polling_mode=polling_mode)
        else:
            worker.init_blocking_progress_mode()

        ep = ucx_api.UCXEndpoint.create(
            worker,
            self.server_address,
            self.port,
            endpoint_error_handling=self.args.error_handling,
        )

        # Wireup before starting to transfer data
        wireup_requests = _transfer_wireup(ep, server=False)
        _wait_requests(worker, self.args.progress_mode, wireup_requests)

        times = []
        contention_metric = None

        async def _transfer():
            nonlocal contention_metric
            if self.args.reuse_alloc:
                recv_msg = Array(xp.zeros(self.args.n_bytes, dtype="u1"))

            if self.args.cuda_profile:
                xp.cuda.profiler.start()
            if self.args.report_gil_contention:
                from gilknocker import KnockKnock

                # Use smallest polling interval
                # possible to ensure, contention will always
                # be zero for small messages otherwise
                # and inconsistent for large messages.
                knocker = KnockKnock(polling_interval_micros=1)
                knocker.start()

            for i in range(self.args.n_iter + self.args.n_warmup_iter):
                start = monotonic()

                if not self.args.reuse_alloc:
                    recv_msg = Array(xp.zeros(self.args.n_bytes, dtype="u1"))

                if self.args.enable_am:
                    send_request = ep.am_send(send_msg)
                    await _wait_requests_checked(
                        worker,
                        self.args.progress_mode,
                        self.args.asyncio_wait,
                        [send_request],
                    )
                    recv_request = ep.am_recv()
                    await _wait_requests_checked(
                        worker,
                        self.args.progress_mode,
                        self.args.asyncio_wait,
                        [recv_request],
                    )
                else:
                    requests = [
                        ep.tag_send(send_msg, tag=ucx_api.UCXXTag(1)),
                        ep.tag_recv(recv_msg, tag=ucx_api.UCXXTag(0)),
                    ]
                    await _wait_requests_checked(
                        worker,
                        self.args.progress_mode,
                        self.args.asyncio_wait,
                        requests,
                    )

                stop = monotonic()
                if i >= self.args.n_warmup_iter:
                    times.append(stop - start)

            if self.args.report_gil_contention:
                knocker.stop()
                contention_metric = knocker.contention_metric
            if self.args.cuda_profile:
                xp.cuda.profiler.stop()

            await _send_terminal_ack(ep, worker, self.args)

        loop = get_event_loop()
        loop.run_until_complete(_transfer())

        self.queue.put(times)
        if self.args.report_gil_contention:
            self.queue.put(contention_metric)

    def print_backend_specific_config(self):
        delay_progress_str = (
            f"True ({self.args.max_outstanding})"
            if self.args.delay_progress is True
            else "False"
        )

        print_key_value(
            key="Transfer API", value=f"{'AM' if self.args.enable_am else 'TAG'}"
        )
        print_key_value(key="Progress mode", value=f"{self.args.progress_mode}")
        print_key_value(key="Asyncio wait", value=f"{self.args.asyncio_wait}")
        print_key_value(key="Delay progress", value=f"{delay_progress_str}")
        print_key_value(key="UCX_TLS", value=f"{ucxx.get_config()['TLS']}")
        print_key_value(
            key="UCX_NET_DEVICES", value=f"{ucxx.get_config()['NET_DEVICES']}"
        )
