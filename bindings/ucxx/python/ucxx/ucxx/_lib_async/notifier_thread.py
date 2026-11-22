# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause


import asyncio
import logging
from concurrent.futures import TimeoutError

import ucxx._lib.libucxx as ucx_api

logger = logging.getLogger("ucx")


async def _run_request_notifier(worker):
    return worker.run_request_notifier()


async def _notifier_coroutine(worker):
    worker.populate_python_futures_pool()
    finished = worker.wait_request_notifier()
    if finished:
        return True

    # Notify all enqueued waiting futures
    await _run_request_notifier(worker)

    return False


def _notifierThread(event_loop, worker, q):
    logger.debug("Starting Notifier Thread")
    asyncio.set_event_loop(event_loop)
    shutdown = False

    while True:
        worker.populate_python_futures_pool()
        state = worker.wait_request_notifier(period_ns=int(1e9))  # 1 second timeout

        if not q.empty():
            q_val = q.get()
            if q_val == "shutdown":
                logger.debug("_notifierThread shutting down")
                shutdown = True
            else:
                logger.warning(
                    f"_notifierThread got unknown message from IPC queue: {q_val}"
                )

        if state == ucx_api.PythonRequestNotifierWaitState.Shutdown or shutdown is True:
            break
        elif state == ucx_api.PythonRequestNotifierWaitState.Timeout:
            continue

        # Notify all enqueued waiting futures
        task = asyncio.run_coroutine_threadsafe(
            _run_request_notifier(worker), event_loop
        )
        try:
            task.result(0.01)
        except TimeoutError:
            # Do NOT cancel the task here. run_request_notifier() sets asyncio
            # Futures for completed UCX requests; cancelling leaves them unset and
            # causes send/recv to hang.
            logger.debug("Notifier Thread Result Timeout (task left scheduled)")
        except Exception as e:
            logger.debug(f"Notifier Thread Result Exception: {e}")

    # Clear all Python futures to ensure no references are held to the
    # `ucxx::Worker` that will prevent destructors from running.
    worker.clear_python_futures_pool()
