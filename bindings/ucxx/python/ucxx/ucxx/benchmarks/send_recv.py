# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import inspect
import multiprocessing as mp
import os

import numpy as np

import ucxx
from ucxx._lib_async.utils import get_event_loop
from ucxx.benchmarks.backends.asyncio import AsyncioClient, AsyncioServer
from ucxx.benchmarks.backends.socket import SocketClient, SocketServer
from ucxx.benchmarks.backends.ucxx_async import (
    UCXPyAsyncClient,
    UCXPyAsyncServer,
)
from ucxx.benchmarks.backends.ucxx_core import UCXPyCoreClient, UCXPyCoreServer
from ucxx.utils import (
    format_bytes,
    parse_bytes,
    print_key_value,
    print_separator,
)

mp = mp.get_context("spawn")


def _get_backend_implementation(backend):
    if backend == "ucxx-async":
        return {"client": UCXPyAsyncClient, "server": UCXPyAsyncServer}
    elif backend == "ucxx-core":
        return {"client": UCXPyCoreClient, "server": UCXPyCoreServer}
    elif backend == "asyncio":
        return {"client": AsyncioClient, "server": AsyncioServer}
    elif backend == "socket":
        return {"client": SocketClient, "server": SocketServer}
    elif backend == "tornado":
        try:
            import tornado  # noqa: F401
        except ImportError as e:
            raise e
        else:
            from ucxx.benchmarks.backends.tornado import (
                TornadoClient,
                TornadoServer,
            )

            return {"client": TornadoClient, "server": TornadoServer}

    raise ValueError(f"Unknown backend {backend}")


def _set_cuda_device(object_type, device):
    if object_type in ["cupy", "rmm"]:
        from ucxx._cuda_context import ensure_cuda_context

        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        ensure_cuda_context(0)


def server(queue, args):
    if args.server_cpu_affinity >= 0:
        os.sched_setaffinity(0, [args.server_cpu_affinity])

    _set_cuda_device(args.object_type, args.server_dev)

    server = _get_backend_implementation(args.backend)["server"](args, queue)

    if inspect.iscoroutinefunction(server.run):
        loop = get_event_loop()
        loop.run_until_complete(server.run())
    else:
        server.run()


def client(queue, port, server_address, args):
    if args.client_cpu_affinity >= 0:
        os.sched_setaffinity(0, [args.client_cpu_affinity])

    _set_cuda_device(args.object_type, args.client_dev)

    client = _get_backend_implementation(args.backend)["client"](
        args, queue, server_address, port
    )

    if inspect.iscoroutinefunction(client.run):
        loop = get_event_loop()
        loop.run_until_complete(client.run())
    else:
        client.run()

    times = queue.get()
    if args.report_gil_contention:
        contention_metric = queue.get()

    assert len(times) == args.n_iter
    bw_avg = format_bytes(2 * args.n_iter * args.n_bytes * args.n_buffers / sum(times))
    bw_med = format_bytes(2 * args.n_bytes * args.n_buffers / np.median(times))
    lat_avg = int(sum(times) * 1e9 / (2 * args.n_iter))
    lat_med = int(np.median(times) * 1e9 / 2)

    print("Roundtrip benchmark")
    print_separator(separator="=")
    print_key_value(key="Iterations", value=f"{args.n_iter}")
    print_key_value(key="Bytes", value=f"{format_bytes(args.n_bytes)}")
    print_key_value(key="Number of buffers", value=f"{args.n_buffers}")
    print_key_value(key="Object type", value=f"{args.object_type}")
    print_key_value(key="Reuse allocation", value=f"{args.reuse_alloc}")
    print_key_value(key="Backend", value=f"{args.backend}")
    client.print_backend_specific_config()
    print_separator(separator="=")
    if args.object_type == "numpy":
        print_key_value(key="Device(s)", value="CPU-only")
        s_aff = (
            args.server_cpu_affinity
            if args.server_cpu_affinity >= 0
            else "affinity not set"
        )
        c_aff = (
            args.client_cpu_affinity
            if args.client_cpu_affinity >= 0
            else "affinity not set"
        )
        print_key_value(key="Server CPU", value=f"{s_aff}")
        print_key_value(key="Client CPU", value=f"{c_aff}")
    else:
        print_key_value(key="Device(s)", value=f"{args.server_dev}, {args.client_dev}")
    print_separator(separator="=")
    print_key_value("Bandwidth (average)", value=f"{bw_avg}/s")
    print_key_value("Bandwidth (median)", value=f"{bw_med}/s")
    print_key_value("Latency (average)", value=f"{lat_avg} ns")
    print_key_value("Latency (median)", value=f"{lat_med} ns")
    if args.report_gil_contention:
        print_key_value("GIL contention", value=f"{contention_metric}")
    if not args.no_detailed_report:
        print_separator(separator="=")
        print_key_value(key="Iterations", value="Bandwidth, Latency")
        print_separator(separator="-")
        for i, t in enumerate(times):
            ts = format_bytes(2 * args.n_bytes * args.n_buffers / t)
            lat = int(t * 1e9 / 2)
            print_key_value(key=i, value=f"{ts}/s, {lat}ns")


def parse_args():
    parser = argparse.ArgumentParser(description="Roundtrip benchmark")
    if callable(parse_bytes):
        parser.add_argument(
            "-n",
            "--n-bytes",
            metavar="BYTES",
            default="10 Mb",
            type=parse_bytes,
            help="Message size. Default '10 Mb'.",
        )
    else:
        parser.add_argument(
            "-n",
            "--n-bytes",
            metavar="BYTES",
            default=10_000_000,
            type=int,
            help="Message size in bytes. Default '10_000_000'.",
        )
    parser.add_argument(
        "-x",
        "--n-buffers",
        default="1",
        type=int,
        help="Number of buffers to transfer using the multi-buffer transfer API. "
        "All buffers will be of same size specified by --n-bytes and same type "
        "specified by --object_type. (default: 1, i.e., single-buffer transfer)",
    )
    parser.add_argument(
        "--n-iter",
        metavar="N",
        default=10,
        type=int,
        help="Number of send / recv iterations (default 10).",
    )
    parser.add_argument(
        "--n-warmup-iter",
        default=10,
        type=int,
        help="Number of send / recv warmup iterations (default 10).",
    )
    parser.add_argument(
        "-b",
        "--server-cpu-affinity",
        metavar="N",
        default=-1,
        type=int,
        help="CPU affinity for server process (default -1: not set).",
    )
    parser.add_argument(
        "-c",
        "--client-cpu-affinity",
        metavar="N",
        default=-1,
        type=int,
        help="CPU affinity for client process (default -1: not set).",
    )
    parser.add_argument(
        "-o",
        "--object_type",
        default="numpy",
        choices=["numpy", "cupy", "rmm"],
        help="In-memory array type.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Whether to print timings per iteration.",
    )
    parser.add_argument(
        "-s",
        "--server-address",
        metavar="ip",
        default=ucxx.utils.get_address(),
        type=str,
        help="Server address (default `ucxx.utils.get_address()`).",
    )
    parser.add_argument(
        "-d",
        "--server-dev",
        metavar="N",
        default=0,
        type=int,
        help="GPU device on server (default 0).",
    )
    parser.add_argument(
        "-e",
        "--client-dev",
        metavar="N",
        default=0,
        type=int,
        help="GPU device on client (default 0).",
    )
    parser.add_argument(
        "--reuse-alloc",
        default=False,
        action="store_true",
        help="Reuse memory allocations between communication.",
    )
    parser.add_argument(
        "--cuda-profile",
        default=False,
        action="store_true",
        help="Setting CUDA profiler.start()/stop() around send/recv "
        "typically used with `nvprof --profile-from-start off "
        "--profile-child-processes`",
    )
    parser.add_argument(
        "--rmm-init-pool-size",
        metavar="BYTES",
        default=None,
        type=int,
        help="Initial RMM pool size (default  1/2 total GPU memory)",
    )
    parser.add_argument(
        "--server-only",
        default=False,
        action="store_true",
        help="Start up only a server process (to be used with --client).",
    )
    parser.add_argument(
        "--client-only",
        default=False,
        action="store_true",
        help="Connect to solitary server process (to be user with --server-only)",
    )
    parser.add_argument(
        "-p",
        "--port",
        default=None,
        help="The port the server will bind to, if not specified, UCX will bind "
        "to a random port. Must be specified when --client-only is used.",
        type=int,
    )
    parser.add_argument(
        "--enable-am",
        default=False,
        action="store_true",
        help="Use the Active Message API instead of TAG for transfers",
    )
    parser.add_argument(
        "--rmm-managed-memory",
        default=False,
        action="store_true",
        help="Use RMM managed memory (requires `--object-type rmm`)",
    )
    parser.add_argument(
        "--no-detailed-report",
        default=False,
        action="store_true",
        help="Disable detailed report per iteration.",
    )
    parser.add_argument(
        "-l",
        "--backend",
        default="ucxx-async",
        type=str,
        help="Backend Library (-l) to use, options are: 'ucxx-async' (default), "
        "'ucxx-core', 'asyncio', 'socket' and 'tornado'.",
    )
    parser.add_argument(
        "--report-gil-contention",
        default=False,
        action="store_true",
        help="Report GIL contention (requires the `gilknocker` package).",
    )
    parser.add_argument(
        "--progress-mode",
        default="thread",
        help="Progress mode for the UCP worker. Valid options are: 'blocking, "
        "'polling', 'thread' and 'thread-polling. (Default: 'thread')'",
        type=str,
    )
    parser.add_argument(
        "--asyncio-wait",
        default=False,
        action="store_true",
        help="Wait for transfer requests with Python's asyncio, requires"
        "`--progress-mode=thread`. (Default: disabled)",
    )
    parser.add_argument(
        "--delay-progress",
        default=False,
        action="store_true",
        help="Only applies to 'ucxx-core' backend: delay ucp_worker_progress calls "
        "until a minimum number of outstanding operations is reached, implies "
        "non-blocking send/recv. The --max-outstanding argument may be used to "
        "control number of maximum outstanding operations. (Default: disabled)",
    )
    parser.add_argument(
        "--max-outstanding",
        metavar="N",
        default=32,
        type=int,
        help="Only applies to 'ucxx-core' backend: number of maximum outstanding "
        "operations, see --delay-progress. (Default: 32)",
    )
    parser.add_argument(
        "--error-handling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable endpoint error handling.",
    )

    args = parser.parse_args()

    if args.cuda_profile and args.object_type == "numpy":
        raise RuntimeError(
            "`--cuda-profile` requires `--object_type=cupy` or `--object_type=rmm`"
        )
    if args.rmm_managed_memory and args.object_type != "rmm":
        raise RuntimeError("`--rmm-managed-memory` requires `--object_type=rmm`")

    backend_impl = _get_backend_implementation(args.backend)
    if not (
        backend_impl["client"].has_cuda_support
        and backend_impl["server"].has_cuda_support
    ):
        if args.object_type in {"cupy", "rmm"}:
            raise RuntimeError(
                f"Backend '{args.backend}' does not support CUDA transfers"
            )

    if args.progress_mode not in ["blocking", "polling", "thread", "thread-polling"]:
        raise RuntimeError(f"Invalid `--progress-mode`: '{args.progress_mode}'")
    if args.asyncio_wait and not args.progress_mode.startswith("thread"):
        raise RuntimeError(
            "`--asyncio-wait` requires `--progress-mode=thread` or "
            "`--progress-mode=thread-polling`"
        )

    if args.n_buffers > 1 and args.backend != "ucxx-async":
        raise RuntimeError(
            "Multi-buffer transfer only support for `--backend=ucxx-async`."
        )

    if args.backend != "ucxx-core" and args.delay_progress:
        raise RuntimeError("`--delay-progress` requires `--backend=ucxx-core`")

    if args.report_gil_contention:
        try:
            import gilknocker  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "Could not import `gilknocker`. Make sure it is installed or "
                "remove the `--report-gil-contention` argument."
            )

    return args


def main():
    args = parse_args()
    server_address = args.server_address

    # if you are the server, only start the `server process`
    # if you are the client, only start the `client process`
    # otherwise, start everything

    if not args.client_only:
        # server process
        q1 = mp.Queue()
        p1 = mp.Process(target=server, args=(q1, args))
        p1.start()
        port = q1.get()
        print(f"Server Running at {server_address}:{port}")
    else:
        port = args.port

    if not args.server_only or args.client_only:
        # client process
        print(f"Client connecting to server at {server_address}:{port}")
        q2 = mp.Queue()
        p2 = mp.Process(target=client, args=(q2, port, server_address, args))
        p2.start()
        p2.join()
        assert not p2.exitcode

    else:
        p1.join()
        assert not p1.exitcode


if __name__ == "__main__":
    main()
