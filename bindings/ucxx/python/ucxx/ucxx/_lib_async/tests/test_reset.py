# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import gc
import weakref

import pytest

import ucxx
from conftest import _CreatedResources
from ucxx._lib_async.listener import _ListenerHandlerTracker
from ucxx._lib_async.utils_test import wait_listener_client_handlers


class ResetAfterN:
    """Calls ucxx.reset() after n calls"""

    def __init__(self, n):
        self.n = n
        self.count = 0

    def __call__(self):
        self.count += 1
        if self.count == self.n:
            ucxx.reset()


class ClosedEndpointHoldingContext:
    """Endpoint-like resource whose transport is dead but context is retained."""

    def __init__(self):
        self.closed = True
        self._ctx = object()

    def abort(self):
        self._ctx = None


@pytest.mark.asyncio
async def test_closed_endpoint_holding_context_is_tracked_and_cleaned_up():
    resources = _CreatedResources()
    endpoint = resources.add(ClosedEndpointHoldingContext())

    assert resources.live_context_owners == ["ClosedEndpointHoldingContext"]

    await resources.close()

    assert endpoint._ctx is None


@pytest.mark.asyncio
async def test_resource_cleanup_times_out_waiting_for_listener_handler():
    resources = _CreatedResources()
    endpoint = ClosedEndpointHoldingContext()
    endpoint._handler_tracker = _ListenerHandlerTracker()
    resources.add(endpoint)

    gate = asyncio.Event()
    endpoint._handler_tracker.submit(gate.wait(), asyncio.get_running_loop())

    with pytest.raises(
        AssertionError,
        match="UCXX listener handlers did not finish within 0.01 seconds",
    ):
        await asyncio.wait_for(resources.wait_for_release(timeout=0.01), timeout=0.1)


@pytest.mark.asyncio
async def test_reset():
    reset = ResetAfterN(2)

    def server(ep):
        ep.abort()
        reset()

    lt = ucxx.create_listener(server)
    ep = await ucxx.create_endpoint(ucxx.get_address(), lt.port)
    await wait_listener_client_handlers(lt)
    lt_ref = weakref.ref(lt)
    ep_ref = weakref.ref(ep)
    del lt
    del ep
    gc.collect()
    assert lt_ref() is None
    assert ep_ref() is None
    reset()


@pytest.mark.asyncio
async def test_reset_waits_for_implicit_resource_release():
    async def release_later(listener, endpoint):
        await asyncio.sleep(0.05)
        assert not listener.closed
        assert not endpoint.closed

    async def server(ep):
        ep.abort()

    listener = ucxx.create_listener(server)
    endpoint = await ucxx.create_endpoint(ucxx.get_address(), listener.port)
    await wait_listener_client_handlers(listener)

    # Deliberately leave open resources owned by a pending task. The autouse
    # fixture must allow ordinary scope-based destruction to finish before reset.
    asyncio.create_task(release_later(listener, endpoint))


@pytest.mark.asyncio
async def test_lt_still_in_scope_error():
    reset = ResetAfterN(2)

    def server(ep):
        ep.abort()
        reset()

    lt = ucxx.create_listener(server)
    ep = await ucxx.create_endpoint(ucxx.get_address(), lt.port)
    await wait_listener_client_handlers(lt)

    del ep
    with pytest.raises(
        ucxx.exceptions.UCXError,
        match="Trying to reset UCX but not all Endpoints and/or Listeners are closed",
    ):
        reset()

    lt.close()


@pytest.mark.asyncio
async def test_ep_still_in_scope_error():
    reset = ResetAfterN(2)

    def server(ep):
        ep.abort()
        reset()

    lt = ucxx.create_listener(server)
    ep = await ucxx.create_endpoint(ucxx.get_address(), lt.port)
    await wait_listener_client_handlers(lt)

    del lt
    with pytest.raises(
        ucxx.exceptions.UCXError,
        match="Trying to reset UCX but not all Endpoints and/or Listeners are closed",
    ):
        reset()

    ep.abort()
