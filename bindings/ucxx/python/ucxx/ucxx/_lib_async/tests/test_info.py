# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import ucxx as ucxx


@pytest.fixture(autouse=True)
def reset():
    ucxx.reset()
    yield
    ucxx.reset()


def test_context_info():
    info = ucxx.get_ucp_context_info()
    assert isinstance(info, str)


def test_worker_info():
    info = ucxx.get_ucp_worker_info()
    assert isinstance(info, str)


@pytest.mark.parametrize(
    "transports",
    ["self", "tcp", "self,tcp"],
)
def test_check_transport(transports):
    transports_list = transports.split(",")
    inactive_transports = list(set(["self", "tcp"]) - set(transports_list))

    ucxx.reset()
    options = {"TLS": transports, "NET_DEVICES": "all"}
    ucxx.init(options)

    active_transports = ucxx.get_active_transports()
    for t in transports_list:
        assert any([at.startswith(t) for at in active_transports])
    for it in inactive_transports:
        assert any([not at.startswith(it) for at in active_transports])
