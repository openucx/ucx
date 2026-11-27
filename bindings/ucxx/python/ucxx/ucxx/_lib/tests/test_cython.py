# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import pytest


@pytest.mark.cython
def test_context_getter():
    from ucxx._lib.tests_cython.test_cython import cython_test_context_getter

    return cython_test_context_getter()
