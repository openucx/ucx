# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import os

import pytest

# Prevent calls such as `cudf = pytest.importorskip("cudf")` from initializing
# a CUDA context. Such calls may cause tests that must initialize the CUDA
# context on the appropriate device to fail.
# For example, without `RAPIDS_NO_INITIALIZE=True`, `test_benchmark_cluster`
# will succeed if running alone, but fails when all tests are run in batch.
os.environ["RAPIDS_NO_INITIALIZE"] = "True"


def pytest_runtest_teardown(item, nextitem):
    gc.collect()


def pytest_addoption(parser):
    parser.addoption(
        "--run-cython", action="store_true", default=False, help="run Cython tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-cython"):
        # --run-cython given in cli: do not skip Cython tests
        return
    skip_cython = pytest.mark.skip(reason="need --run-cython option to run")
    for item in items:
        if "cython" in item.keywords:
            item.add_marker(skip_cython)


def pytest_configure(config: pytest.Config):
    config.addinivalue_line("markers", "cython: mark test as Cython to run")
