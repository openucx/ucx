# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import inspect
import logging
import os
import threading
import traceback
import weakref

import ucxx._lib.libucxx as ucx_api
from ucxx.exceptions import UCXMessageTruncatedError

from .endpoint import Endpoint
from .exchange_peer_info import exchange_peer_info
from .utils import hash64bits

logger = logging.getLogger("ucx")


def _log_exception(message):
    """Log the current exception without retaining traceback frame references."""
    logger.error(
        "%s\n%s",
        message,
        traceback.format_exc().rstrip(),
        stacklevel=2,
    )


class _ListenerHandlerTracker:
    """
    Track client handlers scheduled by a `Listener`.

    Listener callbacks run outside the asyncio event loop and submit their handlers
    with ``run_coroutine_threadsafe``. Tracking the returned futures covers both the
    interval before the coroutine starts and the completion callbacks that run after
    the coroutine body exits.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._futures = set()

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._futures)

    def submit(self, coroutine, event_loop):
        """Submit and track a handler without exposing an untracked interval."""
        try:
            # Keep submission and registration atomic with respect to active_count.
            # The event loop may finish the coroutine before this call returns.
            with self._lock:
                future = asyncio.run_coroutine_threadsafe(coroutine, event_loop)
                self._futures.add(future)
        except BaseException:
            coroutine.close()
            raise

        future.add_done_callback(
            lambda completed: self._handler_done(completed, event_loop)
        )
        return future

    def _handler_done(self, future, event_loop) -> None:
        if not future.cancelled():
            future.exception()

        # run_coroutine_threadsafe marks its concurrent future done from an asyncio
        # Task callback. Retain the future for one more event-loop turn so waiters do
        # not observe an idle listener while that callback still owns the Task.
        try:
            event_loop.call_soon_threadsafe(self._discard, future)
        except RuntimeError:
            # A closed loop cannot retain pending callbacks, and cannot service an
            # asynchronous waiter either.
            self._discard(future)

    def _discard(self, future) -> None:
        with self._lock:
            self._futures.discard(future)

    async def wait(self, progress=None) -> None:
        """Wait until all handlers submitted so far have fully retired."""
        while True:
            with self._lock:
                futures = tuple(self._futures)
            if not futures:
                return
            if progress is not None:
                progress()
                await asyncio.sleep(1e-9)
                continue
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in futures),
                return_exceptions=True,
            )
            await asyncio.sleep(0)


def _finalizer(handler_tracker: _ListenerHandlerTracker) -> None:
    """Listener finalizer.

    If there are active client handlers, log a warning.

    Parameters
    ----------
    handler_tracker: _ListenerHandlerTracker
        Tracks handlers scheduled by this listener.
    """
    active_clients = handler_tracker.active_count
    if active_clients > 0:
        logger.warning(
            f"Listener object is being destroyed, but {active_clients} client "
            "handler(s) is(are) still alive. This usually indicates the Listener "
            "was prematurely destroyed."
        )


class Listener:
    """A handle to the listening service started by `create_listener()`

    The listening continues as long as this object exist or `.close()` is called.
    Please use `create_listener()` to create an Listener.
    """

    def __init__(self, listener, handler_tracker, ctx):
        if not isinstance(listener, ucx_api.UCXListener):
            raise ValueError("listener must be an instance of UCXListener")

        self._listener = listener
        self._handler_tracker = handler_tracker
        # The public Listener owns its context while it is live. The lower-level
        # listener callback receives only a weak reference, preventing stale callback
        # data from extending the context lifetime after this object is released.
        self._ctx = ctx

        weakref.finalize(self, _finalizer, handler_tracker)

    @property
    def closed(self):
        """Is the listener closed?"""
        return self._listener is None

    @property
    def ip(self):
        """The listening network IP address"""
        return self._listener.ip

    @property
    def port(self):
        """The listening network port"""
        return self._listener.port

    @property
    def active_clients(self):
        return self._handler_tracker.active_count

    async def _wait_for_active_clients(self, progress=None):
        await self._handler_tracker.wait(progress=progress)

    def close(self):
        """Closing the listener"""
        self._listener = None
        self._ctx = None


async def _listener_handler_coroutine(
    conn_request,
    ctx_ref,
    func,
    endpoint_error_handling,
    connect_timeout,
):
    # We create the Endpoint in five steps:
    #  1) Create endpoint from conn_request
    #  2) Generate unique IDs to use as tags
    #  3) Exchange endpoint info such as tags
    #  4) Setup control receive callback
    #  5) Execute the listener's callback function
    ctx = ctx_ref()
    if ctx is None:
        logger.debug("ApplicationContext was freed before listener handler started")
        return
    endpoint = None
    ep = None
    try:
        endpoint = conn_request

        seed = os.urandom(16)
        msg_tag = hash64bits("msg_tag", seed, endpoint.handle)

        try:
            peer_info = await exchange_peer_info(
                endpoint=endpoint,
                msg_tag=msg_tag,
                listener=True,
                connect_timeout=connect_timeout,
            )
        except UCXMessageTruncatedError:
            # A truncated message occurs if the remote endpoint closed before
            # exchanging peer info, in that case we should raise the endpoint
            # error instead.
            endpoint.raise_on_error()
            return
        tags = {
            "msg_send": peer_info["msg_tag"],
            "msg_recv": msg_tag,
        }
        ep = Endpoint(endpoint=endpoint, ctx=ctx, tags=tags)

        logger.debug(
            "_listener_handler() server: %s, error handling: %s, msg-tag-send: %s, "
            "msg-tag-recv: %s"
            % (
                hex(endpoint.handle),
                endpoint_error_handling,
                hex(ep._tags["msg_send"]),
                hex(ep._tags["msg_recv"]),
            )
        )

        # Finally, we call `func`
        if inspect.iscoroutinefunction(func):
            try:
                await func(ep)
            except Exception:
                _log_exception("Uncaught listener callback error")
        else:
            func(ep)
    except Exception:
        _log_exception("Unexpected error in listener handler coroutine")
    finally:
        ctx = None
        endpoint = None
        conn_request = None
        ep = None


def _listener_handler(
    conn_request,
    event_loop,
    callback_func,
    ctx_ref,
    endpoint_error_handling,
    connect_timeout,
    handler_tracker,
):
    handler_tracker.submit(
        _listener_handler_coroutine(
            conn_request,
            ctx_ref,
            callback_func,
            endpoint_error_handling,
            connect_timeout,
        ),
        event_loop,
    )
