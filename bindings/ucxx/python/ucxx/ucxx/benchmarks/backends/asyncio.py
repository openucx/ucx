# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import warnings
from time import monotonic

from ucxx.benchmarks.backends.base import BaseClient, BaseServer


class AsyncioServer(BaseServer):
    has_cuda_support = False

    def __init__(self, args, queue):
        self.args = args
        self.queue = queue
        self._serve_task = None

    async def _start_listener(self, port):
        for i in range(10000, 60000):
            try:
                return i, await asyncio.start_server(self.handle_stream, "0.0.0.0", i)
            except OSError:
                continue
        raise Exception("Could not start server")

    async def handle_stream(self, reader, writer):
        for i in range(self.args.n_iter + self.args.n_warmup_iter):
            try:
                recv_msg = await reader.read(self.args.n_bytes)
                writer.write(recv_msg)
                await writer.drain()
            except ConnectionResetError:
                break

        writer.close()
        await writer.wait_closed()

        self._serve_task.cancel()

    async def serve_forever(self):
        if self.args.port is not None:
            port, server = (
                self.args.port,
                await asyncio.start_server(
                    self.handle_stream, "0.0.0.0", self.args.port
                ),
            )
        else:
            port, server = await self._start_listener(None)

        self.queue.put(port)
        async with server:
            await server.serve_forever()

    async def run(self):
        self._serve_task = asyncio.create_task(self.serve_forever())

        try:
            await self._serve_task
        except asyncio.CancelledError:
            pass


class AsyncioClient(BaseClient):
    has_cuda_support = False

    def __init__(self, args, queue, server_address, port):
        self.args = args
        self.queue = queue
        self.server_address = server_address
        self.port = port

    async def run(self):
        reader, writer = await asyncio.open_connection(
            self.server_address, self.port, limit=1024**3
        )

        if self.args.reuse_alloc:
            warnings.warn(
                "Reuse allocation not supported by 'asyncio' backend, it will be "
                "ignored."
            )

        send_msg = ("x" * self.args.n_bytes).encode()

        if self.args.report_gil_contention:
            from gilknocker import KnockKnock

            # Use smallest polling interval possible to ensure, contention will always
            # be zero for small messages otherwise and inconsistent for large messages.
            knocker = KnockKnock(polling_interval_micros=1)
            knocker.start()

        times = []
        for i in range(self.args.n_iter + self.args.n_warmup_iter):
            start = monotonic()

            try:
                writer.write(send_msg)
                await writer.drain()
                await reader.read(self.args.n_bytes)
            except ConnectionResetError:
                break

            stop = monotonic()
            if i >= self.args.n_warmup_iter:
                times.append(stop - start)

        if self.args.report_gil_contention:
            knocker.stop()
        self.queue.put(times)
        if self.args.report_gil_contention:
            self.queue.put(knocker.contention_metric)
        writer.close()
        await writer.wait_closed()
