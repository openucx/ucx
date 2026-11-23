# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import logging
from queue import Empty, Queue

import pytest

import ucxx
from ucxx._lib.libucxx import (
    UCXCanceled,
    UCXCloseError,
    UCXConnectionResetError,
    UCXMessageTruncatedError,
)
from ucxx._lib_async.endpoint import Endpoint
from ucxx._lib_async.utils_test import wait_listener_client_handlers
from ucxx.types import Tag


class _MatchedProbe:
    def __init__(self, tag, message, remove):
        self.matched = True
        self.sender_tag = tag
        self.length = len(message)
        self.handle = object() if remove else None
        self.message = message


class _UnmatchedProbe:
    matched = False
    sender_tag = None
    length = 0
    handle = None


class _CompletedReceive:
    def __init__(self, buffer, message):
        self._buffer = buffer
        self._message = message

    async def wait(self):
        self._buffer.obj[:] = self._message


class _CanceledReceive:
    async def wait(self):
        raise UCXCanceled("Request canceled")


class _TruncatedReceive:
    async def wait(self):
        raise UCXMessageTruncatedError("Message truncated")


class _WorkerWithMatchedMessage:
    def __init__(self, message):
        self.message = message

    def tag_probe(self, tag, remove=False):
        if self.message is None:
            return _UnmatchedProbe()
        probe = _MatchedProbe(tag, self.message, remove)
        if remove:
            self.message = None
        return probe

    def tag_recv_with_handle(self, buffer, probe_result):
        if probe_result.handle is None:
            raise ValueError("TagProbeResult does not own the matched message")
        if buffer.nbytes < probe_result.length:
            return _TruncatedReceive()
        return _CompletedReceive(buffer, probe_result.message)


class _WorkerCancelingMatchedReceive(_WorkerWithMatchedMessage):
    def __init__(self, message):
        super().__init__(message)
        self._receive_attempts = 0

    def tag_recv_with_handle(self, buffer, probe_result):
        self._receive_attempts += 1
        if self._receive_attempts == 1:
            return _CanceledReceive()
        return super().tag_recv_with_handle(buffer, probe_result)


class _ContextWithMatchedMessage:
    def __init__(self, worker):
        self.worker = worker


class _ClosedEndpoint:
    alive = False
    handle = 1

    def raise_on_error(self):
        raise UCXCloseError("Endpoint closed")

    def tag_recv(self, buffer, tag, tag_mask):
        raise AssertionError("receive was submitted on a closed endpoint")


class _EndpointClosingDuringReceive(_ClosedEndpoint):
    alive = True

    def raise_on_error(self):
        pass

    def tag_recv(self, buffer, tag, tag_mask):
        return _CanceledReceive()


class _FailedRequest:
    def __init__(self, status):
        self.status = status


class _FailedMultiReceive:
    status = -25
    requests = (_FailedRequest(0), _FailedRequest(-25), _FailedRequest(0))

    async def wait(self):
        pass

    def check_error(self):
        raise UCXConnectionResetError("Connection reset by remote peer")


class _EndpointWithFailedMultiReceive:
    alive = True
    handle = 1

    def raise_on_error(self):
        pass

    def tag_recv_multi(self, tag, tag_mask):
        return _FailedMultiReceive()


def _endpoint_with_matched_message(low_level_endpoint, worker):
    endpoint = Endpoint.__new__(Endpoint)
    endpoint._ep = low_level_endpoint
    endpoint._ctx = _ContextWithMatchedMessage(worker)
    endpoint._tags = {"msg_recv": Tag(1)}
    endpoint._recv_count = 0
    endpoint._finished_recv_count = 0
    endpoint._close_after_n_recv = None
    return endpoint


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "low_level_endpoint",
    [_ClosedEndpoint(), _EndpointClosingDuringReceive()],
    ids=["already-closed", "closes-during-submission"],
)
async def test_recv_matched_message_after_endpoint_closes(low_level_endpoint):
    message = b"message received before endpoint closed"
    worker = _WorkerWithMatchedMessage(message)
    endpoint = _endpoint_with_matched_message(low_level_endpoint, worker)
    received = bytearray(len(message))

    await endpoint.recv(received)

    assert received == message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "low_level_endpoint",
    [_ClosedEndpoint(), _EndpointClosingDuringReceive()],
    ids=["already-closed", "closes-during-submission"],
)
async def test_recv_matched_message_after_endpoint_closes_buffer_too_small(
    low_level_endpoint,
):
    message = b"message received before endpoint closed"
    worker = _WorkerWithMatchedMessage(message)
    endpoint = _endpoint_with_matched_message(low_level_endpoint, worker)

    with pytest.raises(UCXMessageTruncatedError):
        await endpoint.recv(bytearray(len(message) - 1))


@pytest.mark.asyncio
async def test_recv_worker_cancelation_is_not_retried():
    message = b"message received before endpoint closed"
    worker = _WorkerCancelingMatchedReceive(message)
    endpoint = _endpoint_with_matched_message(_ClosedEndpoint(), worker)

    with pytest.raises(UCXCanceled):
        await endpoint.recv(bytearray(len(message)))


@pytest.mark.asyncio
async def test_recv_multi_logs_aggregate_and_child_statuses(caplog):
    endpoint = _endpoint_with_matched_message(
        _EndpointWithFailedMultiReceive(), _WorkerWithMatchedMessage(None)
    )

    with caplog.at_level(logging.ERROR, logger="ucx"):
        with pytest.raises(UCXConnectionResetError):
            await endpoint.recv_multi()

    assert (
        "recv_multi failed: aggregate status=-25, child statuses=[0, -25, 0]"
        in caplog.messages
    )


@pytest.mark.asyncio
@pytest.mark.flaky(
    reruns=3,
    only_rerun="Trying to reset UCX but not all Endpoints and/or Listeners are closed",
)
@pytest.mark.parametrize("server_close_callback", [True, False])
async def test_close_callback(server_close_callback):
    closed = [False]
    callback_registered = asyncio.Event()

    def _close_callback():
        closed[0] = True

    async def server_node(ep):
        if server_close_callback is True:
            try:
                ep.set_close_callback(_close_callback)
            except RuntimeError:
                # If we fail to set the close callback because the remote endpoint
                # has closed already, simply execute the callback.
                _close_callback()
            callback_registered.set()
            while closed[0] is False:
                await asyncio.sleep(0.01)
        else:
            await callback_registered.wait()
        await ep.close()

    async def client_node(port):
        ep = await ucxx.create_endpoint(
            ucxx.get_address(),
            port,
        )
        if server_close_callback is False:
            try:
                ep.set_close_callback(_close_callback)
            except RuntimeError:
                # If we fail to set the close callback because the remote endpoint
                # has closed already, simply execute the callback.
                _close_callback()
            callback_registered.set()
            while closed[0] is False:
                await asyncio.sleep(0.01)
        else:
            await callback_registered.wait()
        await ep.close()

    listener = ucxx.create_listener(
        server_node,
    )
    await client_node(listener.port)
    await wait_listener_client_handlers(listener)
    while closed[0] is False:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("transfer_api", ["am", "tag", "tag_multi"])
async def test_cancel(transfer_api):
    q = Queue()

    async def server_node(ep):
        while True:
            try:
                # Make sure the listener doesn't return before the client schedules
                # the message to receive. If this is not done, UCXConnectionResetError
                # may be raised instead of UCXCanceledError.
                q.get(timeout=0.01)
                return
            except Empty:
                await asyncio.sleep(0)

    async def client_node(port):
        ep = await ucxx.create_endpoint(ucxx.get_address(), port)
        try:
            if transfer_api == "am":
                _, pending = await asyncio.wait(
                    [asyncio.create_task(ep.am_recv())], timeout=0.001
                )
            elif transfer_api == "tag":
                msg = bytearray(1)
                _, pending = await asyncio.wait(
                    [asyncio.create_task(ep.recv(msg))], timeout=0.001
                )
            else:
                _, pending = await asyncio.wait(
                    [asyncio.create_task(ep.recv_multi())], timeout=0.001
                )

            q.put("close")
            await asyncio.wait(pending)
            (pending,) = pending
            result = pending.result()
            assert isinstance(result, Exception)
            raise result
        except Exception as e:
            await ep.close()
            raise e

    listener = ucxx.create_listener(server_node)
    with pytest.raises(
        ucxx.exceptions.UCXCanceledError,
        # TODO: Add back custom UCXCanceledError messages?
    ):
        await client_node(listener.port)
    await wait_listener_client_handlers(listener)
    listener.close()
