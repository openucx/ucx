# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause


import asyncio
import hashlib
import multiprocessing as mp

mp = mp.get_context("spawn")


def get_event_loop():
    """
    Get running or create new event loop

    In Python 3.10, the behavior of `get_event_loop()` is deprecated and in
    the future it will be an alias of `get_running_loop()`. In several
    situations, UCX-Py needs to create a new event loop, so this function
    will remain for now as an alternative to the behavior of `get_event_loop()`
    from Python < 3.10, returning the `get_running_loop()` if an event loop
    exists, or returning a new one with `new_event_loop()` otherwise.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop()


def hash64bits(*args):
    """64 bit unsigned hash of `args`"""
    # 64 bits hexdigest
    h = hashlib.sha1(bytes(repr(args), "utf-8")).hexdigest()[:16]
    # Convert to an integer and return
    return int(h, 16)
