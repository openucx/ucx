# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
from functools import partial

import pytest

import ucxx
from ucxx._lib_async.utils_test import wait_listener_client_handlers
from ucxx._lib_async.utils_test import recv_close_receipt, send_close_receipt
from ucxx.types import Tag

Message = bytearray(b"0" * 10)
Listener = None


async def _wait_for_probe(probe, is_matched=lambda probe_info: probe_info.matched):
    while not is_matched(probe_info := probe()):
        ucxx.progress()
        await asyncio.sleep(0)
    return probe_info


@pytest.mark.asyncio
async def test_wait_for_probe_yields(monkeypatch):
    """An unmatched probe must yield so another task can make it match."""

    ready = False

    class ProbeInfo:
        matched = False

    def probe():
        nonlocal ready
        if ready:
            return type("ProbeInfo", (), {"matched": True})()
        return ProbeInfo()

    async def make_probe_match():
        nonlocal ready
        await asyncio.sleep(0)
        ready = True

    monkeypatch.setattr(ucxx, "progress", lambda: None)
    ready_task = asyncio.create_task(make_probe_match())
    await _wait_for_probe(probe)
    await ready_task


async def _server_node(ep, listener=None, coroutine=None):
    global Listener

    received = await coroutine(ep)
    assert received == Message

    await send_close_receipt(ep)
    while not ep.closed:
        await asyncio.sleep(0)  # Yield task
    await ep.close()
    Listener.close()


async def _server_node_context_tag_coroutine(ep):
    ctx = ep._ctx
    probe_info = await _wait_for_probe(lambda: ctx.tag_probe(Tag(ep._tags["msg_recv"])))
    assert probe_info.sender_tag == Tag(ep._tags["msg_recv"])
    assert probe_info.length == len(Message)
    assert probe_info.handle is None
    received = bytearray(len(Message))
    await ctx.recv(received, Tag(ep._tags["msg_recv"]))
    return received


async def _server_node_endpoint_tag_coroutine(ep):
    probe_info = await _wait_for_probe(ep.tag_probe)
    assert probe_info.sender_tag == Tag(ep._tags["msg_recv"])
    assert probe_info.length == len(Message)
    assert probe_info.handle is None
    received = bytearray(len(Message))
    await ep.recv(received)
    return received


async def _server_node_context_tag_remove_coroutine(ep):
    ctx = ep._ctx
    probe_info = await _wait_for_probe(
        lambda: ctx.tag_probe(Tag(ep._tags["msg_recv"]), remove=True)
    )
    assert probe_info.sender_tag == Tag(ep._tags["msg_recv"])
    assert probe_info.length == len(Message)
    received = bytearray(len(Message))
    await ctx.recv_with_handle(received, probe_info)
    return received


async def _server_node_endpoint_tag_remove_coroutine(ep):
    probe_info = await _wait_for_probe(lambda: ep.tag_probe(remove=True))
    assert probe_info.sender_tag == Tag(ep._tags["msg_recv"])
    assert probe_info.length == len(Message)
    received = bytearray(len(Message))
    await ep.recv_with_handle(received, probe_info)
    return received


async def _server_node_am_coroutine(ep):
    await _wait_for_probe(ep._ep.am_probe, bool)
    return bytes(await ep.am_recv())


async def _client_node(probe_type, port):
    ep = await ucxx.create_endpoint(
        ucxx.get_address(),
        port,
    )
    if probe_type == "am":
        await ep.am_send(Message)
    elif probe_type in ("tag", "tag_remove"):
        await ep.send(Message)
    await recv_close_receipt(ep)
    await ep.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("api_type", ["endpoint", "context"])
@pytest.mark.parametrize("probe_type", ["am", "tag", "tag_remove"])
async def test_message_probe(api_type, probe_type):
    global Listener

    if api_type == "context" and probe_type == "am":
        pytest.skip("There is no AM probe directly in the context")

    if probe_type == "am":
        server_node_partial = partial(_server_node, coroutine=_server_node_am_coroutine)
    elif probe_type == "tag":
        if api_type == "context":
            server_node_partial = partial(
                _server_node, coroutine=_server_node_context_tag_coroutine
            )
        elif api_type == "endpoint":
            server_node_partial = partial(
                _server_node, coroutine=_server_node_endpoint_tag_coroutine
            )
    elif probe_type == "tag_remove":
        if api_type == "context":
            server_node_partial = partial(
                _server_node, coroutine=_server_node_context_tag_remove_coroutine
            )
        elif api_type == "endpoint":
            server_node_partial = partial(
                _server_node, coroutine=_server_node_endpoint_tag_remove_coroutine
            )

    Listener = ucxx.create_listener(
        server_node_partial,
    )
    await _client_node(probe_type, Listener.port)

    await wait_listener_client_handlers(Listener)
    while not Listener.closed:
        await asyncio.sleep(0.01)
