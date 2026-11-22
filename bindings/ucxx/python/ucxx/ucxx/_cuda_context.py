# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CUDA context management using cuda.core.

Provides helpers to ensure a CUDA context is created and to synchronize
the default stream.
"""

from cuda.core import Device


def ensure_cuda_context(device_id: int = 0) -> None:
    """Ensure a CUDA context exists for the given device and set it as current.

    Parameters
    ----------
    device_id : int, optional
        The CUDA device index (default: 0).
    """
    Device(device_id).set_current()


def synchronize_default_stream(device_id: int = 0) -> None:
    """Synchronize the default CUDA stream of the current device.

    Required when coordinating with UCX CUDA transfers (e.g. before send/recv
    of CUDA buffers).

    Parameters
    ----------
    device_id : int, optional
        The CUDA device index (default: 0).
    """
    device = Device(device_id)
    device.set_current()
    device.sync()
