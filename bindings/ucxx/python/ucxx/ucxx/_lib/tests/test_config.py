# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import os
from unittest.mock import patch

import pytest

import ucxx._lib.libucxx as ucx_api
from ucxx._lib.arr import Array
from ucxx._lib.libucxx import UCXInvalidParamError


def test_config_property():
    # Cache user-defined UCX_TLS and unset it to test default value
    tls = os.environ.get("UCX_TLS", None)
    if tls is not None:
        del os.environ["UCX_TLS"]

    ctx = ucx_api.UCXContext()
    config = ctx.config
    assert isinstance(config, dict)
    assert config["TLS"] == "all"

    # Restore user-defined UCX_TLS
    if tls is not None:
        os.environ["UCX_TLS"] = tls


def test_set_env():
    os.environ["UCX_SEG_SIZE"] = "2M"
    ctx = ucx_api.UCXContext()
    config = ctx.config
    assert config["SEG_SIZE"] == os.environ["UCX_SEG_SIZE"]


def test_init_options():
    os.environ["UCX_SEG_SIZE"] = "2M"  # Should be ignored
    options = {"SEG_SIZE": "3M"}
    ctx = ucx_api.UCXContext(options)
    config = ctx.config
    assert config["SEG_SIZE"] == options["SEG_SIZE"]


@pytest.mark.skipif(
    ucx_api.get_ucx_version() >= (1, 12, 0),
    reason="Beginning with UCX >= 1.12, it's only possible to validate "
    "UCP options but not options from other modules such as UCT. "
    "See https://github.com/openucx/ucx/issues/7519.",
)
def test_init_unknown_option():
    options = {"UNKNOWN_OPTION": "3M"}
    with pytest.raises(UCXInvalidParamError):
        ucx_api.UCXContext(options)


def test_init_invalid_option():
    options = {"SEG_SIZE": "invalid-size"}
    with pytest.raises(UCXInvalidParamError):
        ucx_api.UCXContext(options)


@pytest.mark.parametrize("feature_flag", [ucx_api.Feature.TAG, ucx_api.Feature.STREAM])
def test_feature_flags_mismatch(feature_flag):
    ctx = ucx_api.UCXContext(feature_flags=(feature_flag,))
    worker = ucx_api.UCXWorker(ctx)
    addr = worker.address
    ep = ucx_api.UCXEndpoint.create_from_worker_address(
        worker, addr, endpoint_error_handling=False
    )
    msg = Array(bytearray(10))
    if feature_flag != ucx_api.Feature.TAG:
        with pytest.raises(
            ValueError, match="UCXContext must be created with `Feature.TAG`"
        ):
            ep.tag_send(msg, tag=ucx_api.UCXXTag(0))
        with pytest.raises(
            ValueError, match="UCXContext must be created with `Feature.TAG`"
        ):
            ep.tag_recv(msg, tag=ucx_api.UCXXTag(0))
    if feature_flag != ucx_api.Feature.STREAM:
        with pytest.raises(
            ValueError, match="UCXContext must be created with `Feature.STREAM`"
        ):
            ep.stream_send(msg)
        with pytest.raises(
            ValueError, match="UCXContext must be created with `Feature.STREAM`"
        ):
            ep.stream_recv(msg)


@patch.dict(os.environ, {"UCX_TLS": "^cuda"})
def test_no_cuda_support():
    ctx = ucx_api.UCXContext(feature_flags=(ucx_api.Feature.TAG,))
    assert ctx.cuda_support is False
