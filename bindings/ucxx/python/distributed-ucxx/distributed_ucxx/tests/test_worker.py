# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from distributed import Client, Nanny
from distributed.scheduler import Scheduler
from distributed.worker import Worker

from distributed_ucxx.utils_test import gen_test


@pytest.mark.parametrize("protocol", ["ucx", "ucxx"])
@pytest.mark.parametrize("Worker", [Worker, Nanny])
@gen_test()
async def test_protocol_from_scheduler_address(ucxx_loop, protocol, Worker):
    async with Scheduler(protocol=protocol, dashboard_address=":0") as s:
        assert s.address.startswith(f"{protocol}://")
        async with Worker(s.address) as w:
            assert w.address.startswith(f"{protocol}://")
            async with Client(s.address, asynchronous=True) as c:
                info = c.scheduler_info()
                assert info["address"].startswith(f"{protocol}://")
