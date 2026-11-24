# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import socket
import threading
from time import monotonic

import numpy as np

from ucxx.benchmarks.backends.base import BaseClient, BaseServer


class SocketServer(BaseServer):
    has_cuda_support = False

    def __init__(self, args, queue):
        self.args = args
        self.queue = queue
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def _start_listener(self, server, port):
        host = ""
        if port is not None:
            server.bind((host, port))
        else:
            for i in range(10000, 60000):
                try:
                    server.bind((host, i))
                except OSError:
                    continue
                else:
                    port = i
                    break

        server.listen()
        return port

    def handle_client(self, client_socket):
        args = self.args

        if args.reuse_alloc:
            recv_msg = np.zeros(args.n_bytes, dtype="u1")
            assert recv_msg.nbytes == args.n_bytes

        for _ in range(args.n_iter + args.n_warmup_iter):
            if not args.reuse_alloc:
                recv_msg = np.zeros(args.n_bytes, dtype="u1")

            try:
                client_socket.recv_into(recv_msg.data)
                client_socket.sendall(recv_msg.data)
            except socket.error as e:
                print(e)
                break

        client_socket.close()
        return

    def run(self):
        port = self._start_listener(self.server, self.args.port)
        self.queue.put(port)

        client_socket, addr = self.server.accept()
        threading.Thread(target=self.handle_client, args=(client_socket,)).start()

        self.server.close()


class SocketClient(BaseClient):
    has_cuda_support = False

    def __init__(self, args, queue, server_address, port):
        self.args = args
        self.queue = queue
        self.server_address = server_address
        self.port = port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def run(self) -> bool:
        self.client.connect((self.server_address, self.port))
        send_msg = np.arange(self.args.n_bytes, dtype="u1")
        assert send_msg.nbytes == self.args.n_bytes

        if self.args.reuse_alloc:
            recv_msg = np.zeros(self.args.n_bytes, dtype="u1")
            assert recv_msg.nbytes == self.args.n_bytes

        if self.args.report_gil_contention:
            from gilknocker import KnockKnock

            # Use smallest polling interval possible to ensure, contention will always
            # be zero for small messages otherwise and inconsistent for large messages.
            knocker = KnockKnock(polling_interval_micros=1)
            knocker.start()

        times = []
        for i in range(self.args.n_iter + self.args.n_warmup_iter):
            start = monotonic()

            if not self.args.reuse_alloc:
                recv_msg = np.zeros(self.args.n_bytes, dtype="u1")

            self.client.sendall(send_msg.data)
            self.client.recv_into(recv_msg.data)

            stop = monotonic()
            if i >= self.args.n_warmup_iter:
                times.append(stop - start)

        if self.args.report_gil_contention:
            knocker.stop()
        self.queue.put(times)
        if self.args.report_gil_contention:
            self.queue.put(knocker.contention_metric)
