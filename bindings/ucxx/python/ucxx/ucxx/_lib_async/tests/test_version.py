# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import ucxx


def test_get_ucx_version():
    version = ucxx.get_ucx_version()
    assert isinstance(version, tuple)
    assert len(version) == 3
    # Check UCX isn't initialized
    assert ucxx.core._ctx is None


def test_version_constant():
    assert isinstance(ucxx.__version__, str)


def test_ucx_version_constant():
    assert isinstance(ucxx.__ucx_version__, str)
