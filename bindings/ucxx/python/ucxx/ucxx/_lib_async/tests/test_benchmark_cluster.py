# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import tempfile
from itertools import chain

import numpy as np
import pytest

from ucxx.benchmarks.utils import _run_cluster_server, _run_cluster_workers
from ucxx.testing import join_processes, run_in_subprocess, terminate_process


async def _worker(rank, eps, args):
    futures = []
    # Send my rank to all others
    for ep in eps.values():
        futures.append(ep.send(np.array([rank], dtype="u4")))
    # Recv from all others
    result = np.empty(len(eps.values()), dtype="u4")
    futures += list(ep.recv(result[i : i + 1]) for i, ep in enumerate(eps.values()))

    # Wait for transfers to complete
    await asyncio.gather(*futures)

    # We expect to get the sum of all ranks excluding ours
    expect = sum(range(len(eps) + 1)) - rank
    assert expect == result.sum()


@pytest.mark.asyncio
async def test_benchmark_cluster(n_chunks=1, n_nodes=2, n_workers=2):
    server_file = tempfile.NamedTemporaryFile()

    server, server_ret, server_error_q = _run_cluster_server(
        server_file.name, n_nodes * n_workers, error_wrapper=run_in_subprocess
    )

    # Wait for server to become available
    with open(server_file.name, "r") as f:
        while len(f.read()) == 0:
            pass

    workers_with_errors = list(
        chain.from_iterable(
            _run_cluster_workers(
                server_file.name,
                n_chunks,
                n_workers,
                i,
                _worker,
                error_wrapper=run_in_subprocess,
            )
            for i in range(n_nodes)
        )
    )

    workers = [w for w, _ in workers_with_errors]
    worker_error_qs = [eq for _, eq in workers_with_errors]

    join_processes(workers + [server], timeout=30)
    for worker, error_q in zip(workers, worker_error_qs):
        terminate_process(worker, error_queue=error_q)
    terminate_process(server, error_queue=server_error_q)
