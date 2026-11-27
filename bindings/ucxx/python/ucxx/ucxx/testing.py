# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import queue as _queue_module
import traceback as _traceback_module
import time
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue as _MPQueue
from typing import Any, Callable, Optional, Type, Union


def run_in_subprocess(
    target: Callable[..., Any],
    error_queue: _MPQueue[str],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Run function in a subprocess, forwarding exceptions to an error queue.

    Use this as the ``target`` for ``multiprocessing.Process``, passing the
    same ``error_queue`` to :func:`terminate_process`.  Any unhandled exception
    raised by ``target`` will be serialized as a formatted traceback string and
    placed in ``error_queue`` before being re-raised, so the process still
    exits with a non-zero code.  :func:`terminate_process` then reads from
    that queue and raises a :class:`RuntimeError` that includes the full
    subprocess traceback rather than the generic "Process did not exit cleanly"
    message.

    Parameters
    ----------
    target : callable
        The function to run in the subprocess.
    error_queue : multiprocessing.Queue
        Queue that receives the formatted traceback string on failure.
    *args
        Positional arguments forwarded to ``target``.
    **kwargs
        Keyword arguments forwarded to ``target``.
    """
    try:
        target(*args, **kwargs)
    except Exception:
        error_queue.put(_traceback_module.format_exc())
        raise


def join_processes(
    processes: list[Type[BaseProcess]],
    timeout: Union[float, int],
) -> None:
    """
    Join a list of processes with a combined timeout.

    Join a list of processes with a combined timeout, for each process `join()`
    is called with a timeout equal to the difference of `timeout` and the time
    elapsed since this function was called.

    Parameters
    ----------
    processes:
        The list of processes to be joined.
    timeout: float or integer
        Maximum time to wait for all the processes to be joined.
    """
    start = time.monotonic()
    for p in processes:
        t = timeout - (time.monotonic() - start)
        p.join(timeout=t)


def terminate_process(
    process: Type[BaseProcess],
    kill_wait: Union[float, int] = 3.0,
    error_queue: Optional[_MPQueue[str]] = None,
) -> None:
    """
    Ensure a spawned process is terminated.

    Ensure a spawned process is really terminated to prevent the parent process
    (such as pytest) from freezing upon exit.

    Parameters
    ----------
    process:
        The process to be terminated.
    kill_wait: float or integer
        Maximum time to wait for the kill signal to terminate the process.
    error_queue: multiprocessing.Queue, optional
        When the subprocess was started via :func:`run_in_subprocess`, pass
        the same queue here.  If the process exited with a non-zero code,
        ``terminate_process`` will pull the formatted traceback from the queue
        and include it in the raised :class:`RuntimeError`, making the root
        cause visible without digging into subprocess logs.

    Raises
    ------
    RuntimeError
        If the process terminated with a non-zero exit code.
    ValueError
        If the process was still alive after ``kill_wait`` seconds.
    """
    # Ensure process doesn't remain alive and hangs pytest
    if process.is_alive():
        process.kill()

    start_time = time.monotonic()
    while time.monotonic() - start_time < kill_wait:
        if not process.is_alive():
            break

    if process.is_alive():
        process.close()
    elif process.exitcode != 0:
        msg = f"Process did not exit cleanly (exit code: {process.exitcode})"
        if error_queue is not None:
            try:
                tb = error_queue.get_nowait()
                raise RuntimeError(f"{msg}\n\nSubprocess traceback:\n{tb}")
            except _queue_module.Empty:
                pass
        raise RuntimeError(msg)


def wait_requests(worker, progress_mode, requests):
    if not isinstance(requests, list):
        requests = [requests]

    while not all([r.completed for r in requests]):
        if progress_mode == "blocking":
            worker.progress_worker_event()

    for r in requests:
        r.check_error()
