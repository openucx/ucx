# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from distributed import Nanny, Scheduler
from distributed.utils_test import gen_test
from distributed.worker import Worker


class KeyboardInterruptWorker(Worker):
    """A Worker that raises KeyboardInterrupt almost immediately"""

    async def heartbeat(self):
        def raise_err():
            raise KeyboardInterrupt()

        self.loop.add_callback(raise_err)


@gen_test(timeout=120)
async def test_nanny_closed_by_keyboard_interrupt(ucxx_loop):
    async with Scheduler(protocol="ucx", dashboard_address=":0") as s:
        async with Nanny(
            s.address, nthreads=1, worker_class=KeyboardInterruptWorker
        ) as n:
            await n.process.stopped.wait()
            # Check that the scheduler has been notified about the closed worker
            assert "remove-worker" in str(s.get_events())
