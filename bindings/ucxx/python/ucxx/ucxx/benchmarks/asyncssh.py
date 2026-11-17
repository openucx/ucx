# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import json
import logging
import os
import queue
import sys
from functools import partial

logger = logging.getLogger("ucx.asyncssh")
logger.setLevel(logging.getLevelName(os.getenv("UCXPY_ASYNCSSH_LOG_LEVEL", "WARNING")))

try:
    import asyncssh

    class SSHProc(asyncssh.SSHClientSession):
        def __init__(self, out_queue):
            assert isinstance(out_queue, queue.Queue)
            self.out_queue = out_queue

        def data_received(self, data, datatype):
            logger.debug(f"SSHProc.data_received(): {data=}")
            self.out_queue.put(data)

        def connection_lost(self, exc):
            if exc:
                logger.error(f"SSH session error: {exc}", file=sys.stderr)
            else:
                logger.debug(
                    f"SSH connection terminated successfully {self.out_queue.empty()=}"
                )

    class SSHServerProc(SSHProc):
        address = None
        port = None

        def data_received(self, data, datatype):
            if self.address is None and self.port is None:
                logger.debug(f"SSHServerProc.data_received() address: {data=}")

                server_info = json.loads(data)
                self.address = server_info["address"]
                self.port = server_info["port"]

                self.out_queue.put(server_info)
            else:
                super().data_received(data, datatype)

    async def _run_ssh_cluster(
        args,
        server_host,
        worker_hosts,
        num_workers,
        get_server_command,
        get_worker_command,
    ):
        """
        Run benchmarks in an SSH cluster.

        The results are printed to stdout.

        At the moment, only `ucxx.benchmarks.cudf_merge` is supported.

        Parameters
        ----------
        args: Namespace
            The arguments that were passed to `ucxx.benchmarks.cudf_merge`.
        server_host: str
            String containing hostname or IP address of node where the server
            will run.
        worker_hosts: list
            List of strings containing hostnames or IP addresses of nodes where
            workers will run.
        num_workers: int
        get_server_command: callable
            Function returning the full command that the server node will run.
            Must have signature `get_server_command(args, num_workers)`,
            where:
                - `args` is the parsed `argparse.Namespace` object as parsed by
                  the caller application;
                - `num_workers` number of workers in the entire cluster.
        get_worker_command: callable
            Function returning the full command that each worker node will run.
            Must have signature `get_worker_command(args, num_workers, node_idx)`,
            where:
                - `args` is the parsed `argparse.Namespace` object as parsed by
                  the caller application;
                - `num_workers` number of workers in the entire cluster;
                - `node_idx` index of the node that the process will launch.
        """
        logger.debug(f"{server_host=}, {worker_hosts=}")

        async with asyncssh.connect(server_host, known_hosts=None) as conn:
            server_queue = queue.Queue()
            server_cmd = (get_server_command(args, num_workers, logger=logger),)
            logger.debug(f"[{server_host}] {server_cmd=}")
            server_chan, _ = await conn.create_session(
                partial(SSHServerProc, server_queue),
                server_cmd,
            )

            while True:
                try:
                    server_info = server_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                else:
                    break

            logger.info(f"Server session created {server_info=}")

            workers_conn = await asyncio.gather(
                *[asyncssh.connect(host, known_hosts=None) for host in worker_hosts]
            )

            workers_chan, workers_queue = [], []
            for node_idx, worker_conn in enumerate(workers_conn):
                worker_queue = queue.Queue()
                worker_cmd = get_worker_command(
                    server_info,
                    args,
                    num_workers,
                    node_idx,
                    logger=logger,
                )
                logger.debug(f"[{worker_hosts[node_idx]}] {worker_cmd=}")
                worker_chan, _ = await worker_conn.create_session(
                    partial(SSHProc, worker_queue),
                    worker_cmd,
                )

                workers_chan.append(worker_chan)
                workers_queue.append(worker_queue)

            await asyncio.gather(*[chan.wait_closed() for chan in workers_chan])

            await server_chan.wait_closed()

            while not server_queue.empty():
                print(server_queue.get())

            for i, worker_queue in enumerate(workers_queue):
                if not worker_queue.empty():
                    logger.warning(
                        f"Worker {worker_hosts[i]} stdout wasn't empty. This "
                        "likely indicates errors may have occurred. You may "
                        "run with `UCXPY_ASYNCSSH_LOG_LEVEL=DEBUG` to see the "
                        "full output."
                    )
                    while not worker_queue.empty():
                        logger.debug(worker_queue.get())

    def run_ssh_cluster(
        args,
        server_host,
        worker_hosts,
        num_workers,
        get_server_command,
        get_worker_command,
    ):
        """
        Same as `_run_ssh_cluster()` but running on event loop until completed.
        """
        try:
            asyncio.get_event_loop().run_until_complete(
                _run_ssh_cluster(
                    args,
                    server_host,
                    worker_hosts,
                    num_workers,
                    get_server_command,
                    get_worker_command,
                )
            )
        except (OSError, asyncssh.Error) as exc:
            sys.exit(f"SSH connection failed: {exc}")

except ImportError:
    SSHProc = None
    SSHServerProce = None
    run_ssh_cluster = None
