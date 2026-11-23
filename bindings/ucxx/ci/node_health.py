#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Print a short, report-only health summary of the CI node.

The goal is to make failures caused by degraded CI nodes (e.g., timeouts that only
happen on specific runners) attributable from the job log alone. The script prints
identifying information (GPU UUIDs), GPU PCIe/clock state, CPU/cgroup limits and a
few seconds of micro-benchmarks (CPU, host memory and host<->device bandwidth).

This script never fails the job: every probe is best-effort and the exit code is
always 0. Potential problems are emitted as GitHub Actions ``::warning::``
annotations so they are visible in the job summary.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import time

GPU_QUERY_FIELDS = [
    "index",
    "uuid",
    "name",
    "pci.bus_id",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
    "clocks.sm",
    "clocks.max.sm",
    "clocks.mem",
    "clocks.max.mem",
    "temperature.gpu",
    "power.draw",
    "utilization.gpu",
    "memory.used",
    "memory.total",
]
# Renamed from `clocks_throttle_reasons` in newer drivers, try both.
GPU_EVENT_REASON_FIELDS = [
    "clocks_event_reasons.active",
    "clocks_throttle_reasons.active",
]


def warn(message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::warning title=Node health::{message}", flush=True)
    else:
        print(f"WARNING: {message}", flush=True)


def run(cmd: list[str], timeout: float = 30.0) -> str | None:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=True
        ).stdout.strip()
    except Exception:
        return None


def read_file(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def print_host_info() -> None:
    print(f"Hostname: {socket.gethostname()}")
    print(f"RUNNER_NAME: {os.environ.get('RUNNER_NAME', 'N/A')}")
    print(f"CPUs (online/affinity): {os.cpu_count()}/{len(os.sched_getaffinity(0))}")
    print("Load average (1/5/15 min): {:.2f} {:.2f} {:.2f}".format(*os.getloadavg()))

    cpu_max = read_file("/sys/fs/cgroup/cpu.max")
    if cpu_max is not None:
        print(f"cgroup cpu.max: {cpu_max}")
    cpu_stat = read_file("/sys/fs/cgroup/cpu.stat")
    if cpu_stat is not None:
        throttled = [
            line for line in cpu_stat.splitlines() if line.startswith("nr_throttled")
        ]
        print(f"cgroup cpu.stat: {', '.join(throttled) or 'N/A'}")
    cpu_pressure = read_file("/proc/pressure/cpu")
    if cpu_pressure is not None:
        print(f"CPU pressure: {cpu_pressure.splitlines()[0]}")

    cpuinfo = read_file("/proc/cpuinfo") or ""
    model = next(
        (
            ln.split(":", 1)[1].strip()
            for ln in cpuinfo.splitlines()
            if "model name" in ln
        ),
        "N/A",
    )
    mhz = [
        float(ln.split(":", 1)[1])
        for ln in cpuinfo.splitlines()
        if ln.startswith("cpu MHz")
    ]
    print(f"CPU model: {model}")
    if mhz:
        print(
            f"CPU MHz (min/avg/max): {min(mhz):.0f}/{sum(mhz) / len(mhz):.0f}/"
            f"{max(mhz):.0f}"
        )

    meminfo = read_file("/proc/meminfo") or ""
    for line in meminfo.splitlines():
        if line.startswith(("MemTotal", "MemAvailable")):
            print(line)


def print_gpu_info() -> None:
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found, skipping GPU info")
        return

    out = None
    for reason_field in GPU_EVENT_REASON_FIELDS:
        fields = [*GPU_QUERY_FIELDS, reason_field]
        out = run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ]
        )
        if out:
            break
    if not out:
        print("Failed to query GPU information")
        return

    for line in out.splitlines():
        info = dict(zip(fields, (v.strip() for v in line.split(","))))
        print(f"GPU {info['index']}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        try:
            width, width_max = (
                int(info["pcie.link.width.current"]),
                int(info["pcie.link.width.max"]),
            )
            if width < width_max:
                warn(
                    f"GPU {info['uuid']} PCIe link width x{width} is below its "
                    f"maximum x{width_max}"
                )
        except (KeyError, ValueError):
            pass


def bench_cpu() -> None:
    """Single-threaded pure Python loop, useful to compare host CPU speed."""
    n = 5_000_000
    start = time.perf_counter()
    acc = 0
    for i in range(n):
        acc += i & 0xFF
    elapsed = time.perf_counter() - start
    print(f"CPU single-thread Python loop ({n} iterations): {elapsed:.3f} s")


def bench_host_memory(nbytes: int) -> None:
    try:
        import numpy as np
    except ImportError:
        print("numpy not available, skipping host memory benchmark")
        return
    src = np.ones(nbytes, dtype=np.uint8)
    dst = np.empty_like(src)
    np.copyto(dst, src)  # warmup
    iters = 5
    start = time.perf_counter()
    for _ in range(iters):
        np.copyto(dst, src)
    elapsed = time.perf_counter() - start
    print(
        f"Host memcpy ({nbytes >> 20} MiB): {iters * nbytes / elapsed / 1e9:.2f} GB/s"
    )


def bench_device(nbytes: int) -> None:
    try:
        import cupy
        import numpy as np
    except ImportError:
        print("cupy not available, skipping host<->device benchmarks")
        return
    try:
        stream = cupy.cuda.Stream(non_blocking=True)
        pinned = cupy.cuda.alloc_pinned_memory(nbytes)
        host = np.frombuffer(pinned, dtype="u1", count=nbytes)
        host[:] = 1
        dev = cupy.empty(nbytes, dtype="u1")
        dev2 = cupy.empty(nbytes, dtype="u1")
        iters = 5

        def timed(label, fn):
            fn()  # warmup
            stream.synchronize()
            start = time.perf_counter()
            for _ in range(iters):
                fn()
            stream.synchronize()
            bandwidth = iters * nbytes / (time.perf_counter() - start) / 1e9
            print(f"{label} ({nbytes >> 20} MiB): {bandwidth:.2f} GB/s")

        timed("H2D pinned", lambda: dev.set(host, stream=stream))
        timed("D2H pinned", lambda: dev.get(stream=stream, out=host))
        with stream:
            timed("D2D", lambda: dev2.__setitem__(Ellipsis, dev))
    except Exception as e:
        print(f"Device benchmark failed: {e!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--no-bench", action="store_true", help="Skip micro-benchmarks")
    parser.add_argument(
        "--bench-bytes",
        type=int,
        default=256 << 20,
        help="Buffer size for memory benchmarks (default: 256 MiB)",
    )
    args = parser.parse_args()

    for step in (print_host_info, print_gpu_info):
        try:
            step()
        except Exception as e:
            print(f"{step.__name__} failed: {e!r}")

    if not args.no_bench:
        for step, step_args in (
            (bench_cpu, ()),
            (bench_host_memory, (args.bench_bytes,)),
            (bench_device, (args.bench_bytes,)),
        ):
            try:
                step(*step_args)
            except Exception as e:
                print(f"{step.__name__} failed: {e!r}")


if __name__ == "__main__":
    main()
