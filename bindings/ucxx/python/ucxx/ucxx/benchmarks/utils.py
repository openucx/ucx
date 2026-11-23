# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import inspect
import json
import logging
import multiprocessing as mp
import os
import pickle
import threading
from types import ModuleType

import numpy as np

import ucxx
from ucxx.utils import get_address

logger = logging.getLogger("ucx")


def _ensure_cuda_device(devs, rank):
    from ucxx._cuda_context import ensure_cuda_context

    dev_id = devs[rank % len(devs)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_id)
    logger.debug(f"{dev_id=}, {rank=}")
    ensure_cuda_context(0)


def get_allocator(
    object_type: str, rmm_init_pool_size: int, rmm_managed_memory: bool
) -> ModuleType:
    """
    Initialize and return array-allocator based on arguments passed.

    Parameters
    ----------
    object_type: str
        The type of object the allocator should return. Options are: "numpy", "cupy"
        or "rmm".
    rmm_init_pool_size: int
        If the object type is "rmm" (implies usage of RMM pool), define the initial
        pool size.
    rmm_managed_memory: bool
        If the object type is "rmm", use managed memory if `True`, or default memory
        otherwise.
    Returns
    -------
    A handle to a module, one of ``numpy`` or ``cupy`` (if device memory is requested).
    If the object type is ``rmm``, then ``cupy`` is configured to use RMM as an
    allocator.
    """
    if object_type == "numpy":
        import numpy as xp
    elif object_type == "cupy":
        import cupy as xp
    else:
        import cupy as xp

        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator

        rmm.reinitialize(
            pool_allocator=True,
            managed_memory=rmm_managed_memory,
            initial_pool_size=rmm_init_pool_size,
        )
        xp.cuda.set_allocator(rmm_cupy_allocator)

    return xp


async def send_pickled_msg(ep, obj):
    msg = pickle.dumps(obj)
    await ep.send_obj(msg)


async def recv_pickled_msg(ep):
    msg = await ep.recv_obj()
    return pickle.loads(msg)


def _server_process(
    q,
    server_file,
    n_workers,
    ucx_options_list,
):
    if ucx_options_list is not None:
        ucxx.init(ucx_options_list)
    import sys

    async def run():
        lock = threading.Lock()
        eps = {}
        results = {}

        async def server_handler(ep):
            worker_rank, worker_ip, worker_port = await recv_pickled_msg(ep)
            with lock:
                eps[worker_rank] = (worker_ip, worker_port)

            while len(eps) != n_workers:
                await asyncio.sleep(0.1)

            await send_pickled_msg(ep, eps)

            worker_results = await recv_pickled_msg(ep)
            with lock:
                results[worker_rank] = worker_results

        lf = ucxx.create_listener(server_handler)

        if server_file is None:
            fp = open(sys.stdout.fileno(), mode="w", closefd=False)
        else:
            fp = open(server_file, mode="w")
        with fp:
            json.dump({"address": get_address(), "port": lf.port}, fp)

        while len(results) != n_workers:
            await asyncio.sleep(0.1)

        return results

    loop = asyncio.new_event_loop()
    ret = loop.run_until_complete(run())
    for rank in range(n_workers):
        q.put(ret[rank])

    ucxx.stop_notifier_thread()


def _run_cluster_server(
    server_file,
    n_workers,
    ucx_options_list=None,
    error_wrapper=None,
):
    """
    Create a server that synchronizes workers.

    The server will wait for all `n_workers` to connect and communicate their
    endpoint information, then send the aggregate information to all workers
    so that they will create endpoints to each other, in a fully-connected
    network. Each worker will then communicate its result back to the scheduler
    which will return that result back to the caller.

    Parameters
    ----------
    server_file: str or None
        A string containing the path to a file that will be populated to contain
        the address and port of the server, or `None` to print that information
        to stdout.
    num_workers : int
        Number of workers in the entire network, required to infer when all
        workers have connected and completed.
    ucx_options_list: list of dict
        Options to pass to UCX when initializing workers, one for each worker.
    error_wrapper: callable, optional
        When set (e.g. to :func:`ucxx.testing.run_in_subprocess`), each
        spawned process target is wrapped so that unhandled exceptions are
        forwarded to an error queue. The return value gains an extra element:
        ``(process, result_queue, error_queue)``.

    Returns
    -------
    return : tuple
        ``(process, result_queue)`` when *error_wrapper* is ``None``, or
        ``(process, result_queue, error_queue)`` otherwise.
    """
    ctx = mp.get_context("fork")
    q = ctx.Queue()

    if error_wrapper is not None:
        error_q = ctx.Queue()
        p = ctx.Process(
            target=error_wrapper,
            args=(
                _server_process,
                error_q,
                q,
                server_file,
                n_workers,
                ucx_options_list,
            ),
        )
    else:
        error_q = None
        p = ctx.Process(
            target=_server_process,
            args=(
                q,
                server_file,
                n_workers,
                ucx_options_list,
            ),
        )

    p.start()
    if error_q is not None:
        return p, q, error_q
    return p, q


def run_cluster_server(
    server_file,
    n_workers,
    ucx_options_list=None,
):
    """
    Blocking version of `_run_cluster_server()`.

    Provides same behavior as `_run_cluster_server()`, except that it will join
    processes and thus cause the function to be blocking. It will also combine
    the queue as a list with results for each worker in the `[0..n_workers)` range.
    """
    p, q = _run_cluster_server(
        server_file=server_file,
        n_workers=n_workers,
        ucx_options_list=ucx_options_list,
    )

    # Joining the process if the queue is too large (reproducible for more than
    # 32 workers) causes the process to hang. We join the queue results in a
    # list and return the list instead.
    ret = [q.get() for i in range(n_workers)]

    p.join()
    assert not p.exitcode

    return ret


def _worker_process(
    queue,
    server_info,
    num_node_workers,
    rank,
    ucx_options_list,
    ensure_cuda_device,
    func,
    args,
):
    if ensure_cuda_device is True:
        _ensure_cuda_device(args.devs, rank % num_node_workers)

    if ucx_options_list is not None:
        ucxx.init(ucx_options_list[rank])

    async def run():
        eps = {}

        async def server_handler(ep):
            peer_rank = np.empty((1,), dtype=np.uint64)
            await ep.recv(peer_rank)
            assert peer_rank[0] not in eps
            eps[peer_rank[0]] = ep

        lf = ucxx.create_listener(server_handler)

        logger.debug(f"Sending message info to {server_info=}, {rank=}")
        server_ep = await ucxx.create_endpoint(
            server_info["address"], server_info["port"]
        )
        await send_pickled_msg(server_ep, (rank, get_address(), lf.port))

        logger.debug(f"Receiving network info from server {rank=}")
        workers_info = await recv_pickled_msg(server_ep)
        n_workers = len(workers_info)

        logger.debug(f"Creating endpoints to network {rank=}")
        for i in range(rank + 1, n_workers):
            remote_worker_ip, remote_worker_port = workers_info[i]
            eps[i] = await ucxx.create_endpoint(remote_worker_ip, remote_worker_port)
            await eps[i].send(np.array([rank], dtype=np.uint64))

        while len(eps) != n_workers - 1:
            await asyncio.sleep(0.1)

        logger.debug(f"Running worker {rank=}")

        if inspect.iscoroutinefunction(func):
            results = await func(rank, eps, args)
        else:
            results = func(rank, eps, args)

        await send_pickled_msg(server_ep, results)

    loop = asyncio.new_event_loop()
    ret = loop.run_until_complete(run())
    queue.put(ret)

    ucxx.stop_notifier_thread()


def _run_cluster_workers(
    server_info,
    num_workers,
    num_node_workers,
    node_idx,
    worker_func,
    worker_args=None,
    ucx_options_list=None,
    ensure_cuda_device=False,
    error_wrapper=None,
):
    """
    Create `n_workers` UCX processes that each run `worker_func`.

    Each process will first connect to a server spawned with
    `run_cluster_server()` which will synchronize workers across the nodes.

    This function is non-blocking and the processes created by this function
    call are started but not joined, making this function non-blocking. It's the
    user's responsibility to join all processes in the returned list to ensure
    their completion.

    Parameters
    ----------
    server_info: str or dict
        A string containing the path to a file created by `run_cluster_server()`
        containing the address and port of the server. Alternatively, a
        dictionary containing keys `"address"` and `"port"` may be used the same
        way.
    num_workers : int
        Number of workers in the entire network. Every node must run the same
        number of workers, and thus this value should be equal to
        `node_num_workers * num_cluster_nodes`.
    num_node_workers: int
        Number of workers that this node will run.
    node_idx: int
        Index of the node in the entire cluster, within the range
        `[0..num_cluster_nodes)`. This value is used to calculate the rank
        of each worker. Each node must have a unique index.
    worker_func: callable (can be a coroutine)
        Function that each worker executes.
        Must have signature: `worker(rank, eps, args)` where
            - rank is the worker id
            - eps is a dict of ranks to ucx endpoints
            - args given here as `worker_args`
    worker_args: object
        The argument to pass to `worker_func`.
    ucx_options_list: list of dict
        Options to pass to UCX when initializing workers, one for each worker.
    ensure_cuda_device: bool
        If `True`, sets the `CUDA_VISIBLE_DEVICES` environment variable to match
        the proper CUDA device based on the worker's rank and create the CUDA
        context on the corresponding device before calling `import ucxx` for the
        first time on the newly-spawned worker process, otherwise continues
        without modifying `CUDA_VISIBLE_DEVICES` and creating a CUDA context.
        Please note that having this set to `False` may cause all workers to use
        device 0 and will not ensure proper InfiniBand<->GPU mapping on UCX,
        potentially leading to low performance as GPUDirectRDMA will not be
        active.
    error_wrapper: callable, optional
        When set (e.g. to :func:`ucxx.testing.run_in_subprocess`), each
        spawned process target is wrapped so that unhandled exceptions are
        forwarded to an error queue. The returned list then contains
        ``(process, error_queue)`` tuples instead of bare processes.

    Returns
    -------
    processes : list
        A list of processes (when *error_wrapper* is ``None``) or a list of
        ``(process, error_queue)`` tuples otherwise.
    """

    if isinstance(server_info, str):
        with open(server_info, mode="r") as fp:
            server_info = json.load(fp)
    elif not isinstance(server_info, dict):
        raise ValueError(
            "server_info must be the path to a server file, or a dictionary "
            "with the unpacked values."
        )

    processes = []
    ctx = mp.get_context("fork")
    for worker_num in range(num_node_workers):
        rank = node_idx * num_node_workers + worker_num
        q = ctx.Queue()
        if error_wrapper is not None:
            error_q = ctx.Queue()
            p = ctx.Process(
                target=error_wrapper,
                args=(
                    _worker_process,
                    error_q,
                    q,
                    server_info,
                    num_node_workers,
                    rank,
                    ucx_options_list,
                    ensure_cuda_device,
                    worker_func,
                    worker_args,
                ),
            )
        else:
            error_q = None
            p = ctx.Process(
                target=_worker_process,
                args=(
                    q,
                    server_info,
                    num_node_workers,
                    rank,
                    ucx_options_list,
                    ensure_cuda_device,
                    worker_func,
                    worker_args,
                ),
            )
        p.start()
        if error_q is not None:
            processes.append((p, error_q))
        else:
            processes.append(p)

    return processes


def run_cluster_workers(
    server_info,
    num_workers,
    num_node_workers,
    node_idx,
    worker_func,
    worker_args=None,
    ucx_options_list=None,
    ensure_cuda_device=False,
):
    """
    Blocking version of `_run_cluster_workers()`.

    Provides same behavior as `_run_cluster_workers()`, except that it will join
    processes and thus cause the function to be blocking.
    """
    processes = _run_cluster_workers(
        server_info=server_info,
        num_workers=num_workers,
        num_node_workers=num_node_workers,
        node_idx=node_idx,
        worker_func=worker_func,
        worker_args=worker_args,
        ucx_options_list=ucx_options_list,
        ensure_cuda_device=ensure_cuda_device,
    )

    for proc in processes:
        proc.join()
        assert not proc.exitcode
