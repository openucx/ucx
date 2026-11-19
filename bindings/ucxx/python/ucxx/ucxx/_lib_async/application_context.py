# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
import threading
import weakref
from queue import Queue

import ucxx._lib.libucxx as ucx_api
from ucxx._lib.arr import Array
from ucxx.exceptions import UCXMessageTruncatedError
from ucxx.types import Tag

from .continuous_ucx_progress import BlockingMode, PollingMode, ThreadMode
from .endpoint import Endpoint
from .exchange_peer_info import exchange_peer_info
from .listener import Listener, _ListenerHandlerTracker, _listener_handler
from .notifier_thread import _notifierThread
from .utils import get_event_loop, hash64bits

logger = logging.getLogger("ucx")


ProgressTasks = dict()


def clear_progress_tasks():
    global ProgressTasks
    ProgressTasks.clear()


class ApplicationContext:
    """
    The context of the Asyncio interface of UCX.
    """

    _progress_mode = None
    _enable_delayed_submission = None
    _enable_python_future = None

    def __init__(
        self,
        config_dict={},
        progress_mode=None,
        enable_delayed_submission=None,
        enable_python_future=None,
        connect_timeout=None,
    ):
        self.notifier_thread_q = None
        self.notifier_thread = None
        self.progress_mode = progress_mode
        self.enable_delayed_submission = enable_delayed_submission
        self.enable_python_future = enable_python_future

        if connect_timeout is None:
            self.connect_timeout = float(os.environ.get("UCXPY_CONNECT_TIMEOUT", 5))
        else:
            self.connect_timeout = connect_timeout

        # For now, a application context only has one worker
        self.context = ucx_api.UCXContext(config_dict)
        self.worker = ucx_api.UCXWorker(
            self.context,
            enable_delayed_submission=self.enable_delayed_submission,
            enable_python_future=self.enable_python_future,
        )

        self.start_notifier_thread()

        weakref.finalize(self, clear_progress_tasks)

        # Ensure progress even before Endpoints get created, for example to
        # receive messages directly on a worker after a remote endpoint
        # connected with `create_endpoint_from_worker_address`.
        self.continuous_ucx_progress()

    @property
    def progress_mode(self):
        return self._progress_mode

    @progress_mode.setter
    def progress_mode(self, progress_mode):
        if self._progress_mode is None:
            if progress_mode is None:
                if "UCXPY_PROGRESS_MODE" in os.environ:
                    progress_mode = os.environ["UCXPY_PROGRESS_MODE"]
                else:
                    progress_mode = "thread"

            valid_progress_modes = ["blocking", "polling", "thread", "thread-polling"]
            if not isinstance(progress_mode, str) or not any(
                progress_mode == m for m in valid_progress_modes
            ):
                raise ValueError(
                    f"Unknown progress mode '{progress_mode}', valid modes are: "
                    "'blocking', 'polling', 'thread' or 'thread-polling'"
                )

            self._progress_mode = progress_mode
        else:
            raise RuntimeError("Progress mode already set, modifying not allowed")

    @property
    def enable_delayed_submission(self):
        return self._enable_delayed_submission

    @enable_delayed_submission.setter
    def enable_delayed_submission(self, enable_delayed_submission):
        if self._enable_delayed_submission is None:
            if enable_delayed_submission is None:
                if "UCXPY_ENABLE_DELAYED_SUBMISSION" in os.environ:
                    explicit_enable_delayed_submission = (
                        False
                        if os.environ["UCXPY_ENABLE_DELAYED_SUBMISSION"] == "0"
                        else True
                    )
                else:
                    explicit_enable_delayed_submission = self.progress_mode.startswith(
                        "thread"
                    )
            else:
                explicit_enable_delayed_submission = enable_delayed_submission

            if (
                not self.progress_mode.startswith("thread")
                and explicit_enable_delayed_submission
            ):
                raise ValueError(
                    f"Delayed submission requested, but '{self.progress_mode}' does "
                    "not support it, 'thread' or 'thread-polling' progress mode "
                    "required."
                )

            self._enable_delayed_submission = explicit_enable_delayed_submission
        else:
            raise RuntimeError(
                "Enable delayed submission already set, modifying not allowed"
            )

    @property
    def enable_python_future(self):
        return self._enable_python_future

    @enable_python_future.setter
    def enable_python_future(self, enable_python_future):
        if self._enable_python_future is None:
            if enable_python_future is None:
                if "UCXPY_ENABLE_PYTHON_FUTURE" in os.environ:
                    explicit_enable_python_future = (
                        os.environ["UCXPY_ENABLE_PYTHON_FUTURE"] != "0"
                    )
                else:
                    explicit_enable_python_future = False
            else:
                explicit_enable_python_future = enable_python_future

            if (
                not self.progress_mode.startswith("thread")
                and explicit_enable_python_future
            ):
                logger.warning(
                    f"Notifier thread requested, but '{self.progress_mode}' does not "
                    "support it, using Python wait_yield()."
                )
                explicit_enable_python_future = False

            self._enable_python_future = explicit_enable_python_future
        else:
            raise RuntimeError(
                "Enable Python future already set, modifying not allowed"
            )

    @property
    def config(self):
        """UCX configuration options as a dict."""
        return self.context.config

    @property
    def ucp_context_info(self):
        """Low-level UCX info about this endpoint as a string."""
        return self.context.info

    @property
    def ucp_worker(self):
        """The underlying UCP worker handle (ucp_worker_h) as a Python integer."""
        return self.worker.handle

    @property
    def ucxx_worker(self):
        """The underlying UCXX worker pointer (ucxx::Worker*) as a Python integer."""
        return self.worker.ucxx_ptr

    @property
    def ucp_worker_info(self):
        """Return low-level UCX info about this endpoint as a string."""
        return self.worker.info

    @property
    def worker_address(self):
        return self.worker.address

    def clear_progress_tasks(self) -> None:
        global ProgressTasks
        ProgressTasks.clear()

    def start_notifier_thread(self):
        if self.worker.enable_python_future and self.notifier_thread is None:
            logger.debug("UCXX_ENABLE_PYTHON available, enabling notifier thread")
            loop = get_event_loop()
            self.notifier_thread_q = Queue()
            self.notifier_thread = threading.Thread(
                target=_notifierThread,
                args=(loop, self.worker, self.notifier_thread_q),
                name="UCX-Py Async Notifier Thread",
            )
            self.notifier_thread.start()
        else:
            logger.debug(
                "UCXX not compiled with UCXX_ENABLE_PYTHON, disabling notifier thread"
            )

    def stop_notifier_thread(self):
        """
        Stop Python future notifier thread

        Stop the notifier thread if context is running with Python future
        notification enabled via `UCXPY_ENABLE_PYTHON_FUTURE=1` or
        `ucxx.init(..., enable_python_future=True)`.

        .. warning:: When the notifier thread is enabled it may be necessary to
                     explicitly call this method before shutting down the process or
                     or application, otherwise it may block indefinitely waiting for
                     the thread to terminate. Executing `ucxx.reset()` will also run
                     this method, so it's not necessary to have both.
        """
        if self.notifier_thread_q and self.notifier_thread:
            self.notifier_thread_q.put("shutdown")
            while True:
                # Having a timeout is required. During the notifier thread shutdown
                # it may require the GIL, which will cause a deadlock with the `join()`
                # call otherwise.
                self.notifier_thread.join(timeout=0.01)
                if not self.notifier_thread.is_alive():
                    self.notifier_thread = None
                    break
            logger.debug("Notifier thread stopped")
        else:
            logger.debug("Notifier thread not running")

    def create_listener(
        self,
        callback_func,
        port=0,
        endpoint_error_handling=True,
        connect_timeout=5.0,
    ):
        """Create and start a listener to accept incoming connections

        callback_func is the function or coroutine that takes one
        argument -- the Endpoint connected to the client.

        Notice, the listening is closed when the returned Listener
        goes out of scope thus remember to keep a reference to the object.

        Parameters
        ----------
        callback_func: function or coroutine
            A callback function that gets invoked when an incoming
            connection is accepted
        port: int, optional
            An unused port number for listening, or `0` to let UCX assign
            an unused port.
        endpoint_error_handling: boolean, optional
            If `True` (default) enable endpoint error handling raising
            exceptions when an error occurs, may incur in performance penalties
            but prevents a process from terminating unexpectedly that may
            happen when disabled. If `False` endpoint endpoint error handling
            is disabled.
        connect_timeout: float
            Timeout in seconds for exchanging peer info. In some cases, exchanging
            peer information may hang indefinitely, a timeout prevents that. If the
            chosen value is too high it may cause the operation to be stuck for too
            long rather than quickly raising a `TimeoutError` that may be recovered
            from by the application, but under high-load a higher timeout may
            be helpful to prevent exchanging peer info from failing too fast.

        Returns
        -------
        Listener
            The new listener. When this object is deleted, the listening stops
        """
        self.continuous_ucx_progress()
        if port is None:
            port = 0

        loop = get_event_loop()

        logger.info("create_listener() - Start listening on port %d" % port)
        handler_tracker = _ListenerHandlerTracker()
        ret = Listener(
            ucx_api.UCXListener.create(
                worker=self.worker,
                port=port,
                cb_func=_listener_handler,
                cb_args=(
                    loop,
                    callback_func,
                    weakref.ref(self),
                    endpoint_error_handling,
                    connect_timeout,
                    handler_tracker,
                ),
                deliver_endpoint=True,
            ),
            handler_tracker,
            self,
        )
        return ret

    async def create_endpoint(
        self,
        ip_address,
        port,
        endpoint_error_handling=True,
        connect_timeout=5.0,
    ):
        """Create a new endpoint to a server

        Parameters
        ----------
        ip_address: str
            IP address of the server the endpoint should connect to
        port: int
            IP address of the server the endpoint should connect to
        endpoint_error_handling: boolean, optional
            If `True` (default) enable endpoint error handling raising
            exceptions when an error occurs, may incur in performance penalties
            but prevents a process from terminating unexpectedly that may
            happen when disabled. If `False` endpoint endpoint error handling
            is disabled.
        connect_timeout: float
            Timeout in seconds for exchanging peer info. In some cases, exchanging
            peer information may hang indefinitely, a timeout prevents that. If the
            chosen value is too high it may cause the operation to be stuck for too
            long rather than quickly raising a `TimeoutError` that may be recovered
            from by the application, but under high-load a higher timeout may
            be helpful to prevent exchanging peer info from failing too fast.

        Returns
        -------
        Endpoint
            The new endpoint
        """
        self.continuous_ucx_progress()

        ucx_ep = ucx_api.UCXEndpoint.create(
            self.worker, ip_address, port, endpoint_error_handling
        )
        if not self.progress_mode.startswith("thread"):
            self.worker.progress()

        # We create the Endpoint in three steps:
        #  1) Generate unique IDs to use as tags
        #  2) Exchange endpoint info such as tags
        #  3) Use the info to create an endpoint
        seed = os.urandom(16)
        msg_tag = hash64bits("msg_tag", seed, ucx_ep.handle)
        try:
            peer_info = await exchange_peer_info(
                endpoint=ucx_ep,
                msg_tag=msg_tag,
                listener=False,
                connect_timeout=connect_timeout,
            )
        except UCXMessageTruncatedError as e:
            # A truncated message occurs if the remote endpoint closed before
            # exchanging peer info, in that case we should raise the endpoint
            # error, if available.
            ucx_ep.raise_on_error()
            # If no endpoint error is available, re-raise exception.
            raise e

        tags = {
            "msg_send": peer_info["msg_tag"],
            "msg_recv": msg_tag,
        }
        ep = Endpoint(endpoint=ucx_ep, ctx=self, tags=tags)

        logger.debug(
            "create_endpoint() client: %s, error handling: %s, msg-tag-send: %s, "
            "msg-tag-recv: %s"
            % (
                hex(ep._ep.handle),
                endpoint_error_handling,
                hex(ep._tags["msg_send"]),
                hex(ep._tags["msg_recv"]),
            )
        )

        return ep

    async def create_endpoint_from_worker_address(
        self,
        address,
        endpoint_error_handling=True,
    ):
        """Create a new endpoint to a server

        Parameters
        ----------
        address: UCXAddress
        endpoint_error_handling: boolean, optional
            If `True` (default) enable endpoint error handling raising
            exceptions when an error occurs, may incur in performance penalties
            but prevents a process from terminating unexpectedly that may
            happen when disabled. If `False` endpoint endpoint error handling
            is disabled.

        Returns
        -------
        Endpoint
            The new endpoint
        """
        self.continuous_ucx_progress()

        ucx_ep = ucx_api.UCXEndpoint.create_from_worker_address(
            self.worker,
            address,
            endpoint_error_handling,
        )
        if not self.progress_mode.startswith("thread"):
            self.worker.progress()

        ep = Endpoint(endpoint=ucx_ep, ctx=self, tags=None)

        logger.debug(
            "create_endpoint() client: %s, error handling: %s"
            % (hex(ep._ep.handle), endpoint_error_handling)
        )

        return ep

    def continuous_ucx_progress(self, event_loop=None):
        """Guarantees continuous UCX progress

        Use this function to associate UCX progress with an event loop.
        Notice, multiple event loops can be associate with UCX progress.

        This function is automatically called when calling
        `create_listener()` or `create_endpoint()`.

        Parameters
        ----------
        event_loop: asyncio.event_loop, optional
            The event loop to evoke UCX progress. If None,
            `asyncio.get_event_loop()` (`asyncio.new_event_loop()` in
            Python 3.10+) is used.
        """
        loop = event_loop if event_loop is not None else get_event_loop()
        global ProgressTasks
        if loop in ProgressTasks:
            return  # Progress has already been guaranteed for the current event loop

        logger.info(f"Starting progress in '{self.progress_mode}' mode")

        if self.progress_mode == "thread":
            task = ThreadMode(self.worker, loop, polling_mode=False)
        elif self.progress_mode == "thread-polling":
            task = ThreadMode(self.worker, loop, polling_mode=True)
        elif self.progress_mode == "polling":
            task = PollingMode(self.worker, loop)
        elif self.progress_mode == "blocking":
            task = BlockingMode(self.worker, loop)

        ProgressTasks[loop] = task

    def tag_probe(self, tag, remove=False):
        """Probe for tag messages directly on worker without a local Endpoint.

        This method checks if a message with the specified tag is available
        without actually receiving it. This is useful for non-blocking
        message checking.

        Parameters
        ----------
        tag: hashable
            Set a tag that must match the received message.
        remove: bool
            If true, remove the message from the queue and return a
            message handle for efficient reception. If false, leave
            the message in the queue.

        Returns
        -------
        TagProbeResult
            A result object containing:
            - matched: bool indicating if a message was found
            - sender_tag: int sender tag (when matched=True)
            - length: int message length in bytes (when matched=True)
            - handle: int message handle for efficient reception (when matched=True and
                      remove=True)
        """
        if not isinstance(tag, Tag):
            tag = Tag(tag)

        return self.worker.tag_probe(tag, remove=remove)

    # @ucx_api.nvtx_annotate("UCXPY_WORKER_RECV", color="red", domain="ucxpy")
    async def recv(self, buffer, tag):
        """Receive directly on worker without a local Endpoint into `buffer`.

        Parameters
        ----------
        buffer: exposing the buffer protocol or array/cuda interface
            The buffer to receive into. Raise ValueError if buffer
            is smaller than nbytes or read-only.
        tag: hashable, optional
            Set a tag that must match the received message.
        """
        if not isinstance(buffer, Array):
            buffer = Array(buffer)
        if not isinstance(tag, Tag):
            tag = Tag(tag)
        nbytes = buffer.nbytes
        log = "[Worker Recv] worker: %s, tag: %s, nbytes: %d, type: %s" % (
            hex(self.worker.handle),
            hex(tag.value),
            nbytes,
            type(buffer.obj),
        )
        logger.debug(log)

        req = self.worker.tag_recv(buffer, tag)
        return await req.wait()

    async def recv_with_handle(self, buffer, probe_result):
        """Receive tag message using message handle obtained from tag_probe.

        This is more efficient than regular recv as it doesn't need to go through
        the message matching queue again.

        Parameters
        ----------
        buffer: exposing the buffer protocol or array/cuda interface
            The buffer to receive into. Raise ValueError if buffer
            is smaller than nbytes or read-only.
        probe_result: TagProbeResult
            The probe result obtained from tag_probe with remove=True.

        Returns
        -------
        The received data
        """
        if not isinstance(buffer, Array):
            buffer = Array(buffer)

        nbytes = buffer.nbytes
        log = "[Worker Recv With Handle] worker: %s, nbytes: %d, type: %s" % (
            hex(self.worker.handle),
            nbytes,
            type(buffer.obj),
        )
        logger.debug(log)

        req = self.worker.tag_recv_with_handle(buffer, probe_result)
        return await req.wait()
