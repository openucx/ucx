# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from distributed.comm.tests.test_comms import check_deserialize

from distributed_ucxx.utils_test import gen_test


@gen_test()
async def test_ucxx_deserialize(ucxx_loop):
    # Note we see this error on some systems with this test:
    # `socket.gaierror: [Errno -5] No address associated with hostname`
    # This may be due to a system configuration issue.

    await check_deserialize("ucxx://")
