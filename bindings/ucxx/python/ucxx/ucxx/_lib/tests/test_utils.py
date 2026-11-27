# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing
import re
import time
from multiprocessing.queues import Empty

import pytest

from ucxx.testing import join_processes, run_in_subprocess, terminate_process


def _test_process(queue):
    while True:
        try:
            message = queue.get_nowait()
            assert message == "terminate"
            return
        except Empty:
            pass


def _subprocess_target_success(result_queue):
    result_queue.put("done")


def _subprocess_target_failure():
    raise RuntimeError("subprocess raised")


@pytest.mark.parametrize("mp_context", ["default", "fork", "forkserver", "spawn"])
def test_terminate_process_clean(mp_context):
    mp = (
        multiprocessing
        if mp_context == "default"
        else multiprocessing.get_context(mp_context)
    )

    queue = mp.Queue()
    proc = mp.Process(
        target=_test_process,
        args=(queue,),
    )
    proc.start()

    queue.put("terminate")

    proc.join()
    terminate_process(proc)


@pytest.mark.parametrize("mp_context", ["default", "fork", "forkserver", "spawn"])
def test_terminate_process_join_timeout(mp_context):
    mp = (
        multiprocessing
        if mp_context == "default"
        else multiprocessing.get_context(mp_context)
    )

    queue = mp.Queue()
    proc = mp.Process(
        target=_test_process,
        args=(queue,),
    )

    proc.start()
    proc.join(timeout=1)

    with pytest.raises(
        RuntimeError, match=re.escape("Process did not exit cleanly (exit code: -9)")
    ):
        terminate_process(proc)


@pytest.mark.parametrize("mp_context", ["default", "fork", "forkserver", "spawn"])
def test_terminate_process_kill_timeout(mp_context):
    mp = (
        multiprocessing
        if mp_context == "default"
        else multiprocessing.get_context(mp_context)
    )

    queue = mp.Queue()
    proc = mp.Process(
        target=_test_process,
        args=(queue,),
    )
    proc.start()
    proc.join(timeout=1)

    with pytest.raises(
        ValueError, match="Cannot close a process while it is still running.*"
    ):
        terminate_process(proc, kill_wait=0.0)


@pytest.mark.parametrize("mp_context", ["default", "fork", "forkserver", "spawn"])
@pytest.mark.parametrize("num_processes", [1, 2, 4])
def test_join_processes(mp_context, num_processes):
    mp = (
        multiprocessing
        if mp_context == "default"
        else multiprocessing.get_context(mp_context)
    )

    queue = mp.Queue()
    processes = []
    for _ in range(num_processes):
        proc = mp.Process(
            target=_test_process,
            args=(queue,),
        )
        proc.start()
        processes.append(proc)

    start = time.monotonic()
    join_processes(processes, timeout=1.25)
    total_time = time.monotonic() - start
    assert total_time >= 1.25 and total_time < 2.5

    for proc in processes:
        try:
            terminate_process(proc)
        except RuntimeError:
            # The process has to be killed and that will raise a `RuntimeError`
            pass


@pytest.mark.parametrize("mp_context", ["default", "fork", "forkserver", "spawn"])
def test_run_in_subprocess_success(mp_context):
    mp = (
        multiprocessing
        if mp_context == "default"
        else multiprocessing.get_context(mp_context)
    )

    result_q = mp.Queue()
    error_q = mp.Queue()
    proc = mp.Process(
        target=run_in_subprocess,
        args=(_subprocess_target_success, error_q, result_q),
    )
    proc.start()
    proc.join()

    assert proc.exitcode == 0
    assert result_q.get_nowait() == "done"
    with pytest.raises(Empty):
        error_q.get_nowait()


@pytest.mark.parametrize("mp_context", ["default", "fork", "forkserver", "spawn"])
def test_run_in_subprocess_failure(mp_context):
    mp = (
        multiprocessing
        if mp_context == "default"
        else multiprocessing.get_context(mp_context)
    )

    error_q = mp.Queue()
    proc = mp.Process(
        target=run_in_subprocess,
        args=(_subprocess_target_failure, error_q),
    )
    proc.start()
    proc.join()

    assert proc.exitcode != 0
    tb = error_q.get_nowait()
    assert "RuntimeError" in tb
    assert "subprocess raised" in tb


@pytest.mark.parametrize("mp_context", ["default", "fork", "forkserver", "spawn"])
def test_run_in_subprocess_failure_terminate_surfaces_traceback(mp_context):
    mp = (
        multiprocessing
        if mp_context == "default"
        else multiprocessing.get_context(mp_context)
    )

    error_q = mp.Queue()
    proc = mp.Process(
        target=run_in_subprocess,
        args=(_subprocess_target_failure, error_q),
    )
    proc.start()
    proc.join()

    with pytest.raises(
        RuntimeError,
        match=(
            r"(?s)Process did not exit cleanly.*Subprocess traceback:"
            r".*RuntimeError.*subprocess raised"
        ),
    ):
        terminate_process(proc, error_queue=error_q)
