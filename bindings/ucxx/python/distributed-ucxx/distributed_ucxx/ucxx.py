# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
:ref:`UCX`_ based communications for distributed.

See :ref:`communications` for more.

.. _UCX: https://github.com/openucx/ucx
"""

from __future__ import annotations

import asyncio
import functools
import gc
import itertools
import logging
import os
import struct
import weakref
from collections.abc import Awaitable, Callable, Collection
from enum import Enum
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from dask.utils import parse_bytes
from distributed.comm.addressing import parse_host_port, unparse_host_port
from distributed.comm.core import Comm, CommClosedError, Connector, Listener
from distributed.comm.registry import Backend
from distributed.comm.utils import ensure_concrete_host, from_frames, to_frames
from distributed.diagnostics.nvml import (
    CudaDeviceInfo,
    get_device_index_and_uuid,
    has_cuda_context,
)
from distributed.protocol.utils import host_array
from distributed.utils import ensure_ip, get_ip, get_ipv6, log_errors, nbytes

from .config import get_rmm_config, get_ucx_config, setup_config

logger = logging.getLogger(__name__)

# In order to avoid double init when forking/spawning new processes (multiprocess),
# we make sure only to import and initialize UCXX once at first use. This is also
# required to ensure Dask configuration gets propagated to UCXX, which needs
# variables to be set before being imported.
if TYPE_CHECKING:
    try:
        import ucxx
    except ImportError:
        pass
else:
    ucxx = None

device_array = None
pre_existing_cuda_context = False
cuda_context_created = False
multi_buffer = None


_warning_suffix = (
    "This is often the result of a CUDA-enabled library calling a CUDA runtime "
    "function before Dask-CUDA can spawn worker processes. Please make sure any such "
    "function calls don't happen at import time or in the global scope of a program."
)


def _get_device_and_uuid_str(device_info: CudaDeviceInfo) -> str:
    return f"{device_info.device_index} ({str(device_info.uuid)})"


def _warn_existing_cuda_context(device_info: CudaDeviceInfo, pid: int) -> None:
    device_uuid_str = _get_device_and_uuid_str(device_info)
    logger.warning(
        f"A CUDA context for device {device_uuid_str} already exists "
        f"on process ID {pid}. {_warning_suffix}"
    )


def _warn_cuda_context_wrong_device(
    device_info_expected: CudaDeviceInfo, device_info_actual: CudaDeviceInfo, pid: int
) -> None:
    expected_device_uuid_str = _get_device_and_uuid_str(device_info_expected)
    actual_device_uuid_str = _get_device_and_uuid_str(device_info_actual)
    logger.warning(
        f"Worker with process ID {pid} should have a CUDA context assigned to device "
        f"{expected_device_uuid_str}, but instead the CUDA context is on device "
        f"{actual_device_uuid_str}. {_warning_suffix}"
    )


class CudaStream(Enum):
    Default = 0


def synchronize_stream(stream: CudaStream = CudaStream.Default):
    from ucxx._cuda_context import synchronize_default_stream

    if stream == CudaStream.Default:
        synchronize_default_stream()
    else:
        raise ValueError("Unsupported stream")


class gc_disabled:
    def __enter__(self):
        self.was_enabled = gc.isenabled()
        if self.was_enabled:
            gc.disable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.was_enabled:
            gc.enable()


def make_register():
    count = itertools.count()

    def register() -> int:
        """Register a Dask resource with the UCXX context.

        Register a Dask resource with the UCXX context and keep track of it with the
        use of a unique ID for the resource. The resource ID is later used to
        deregister the resource from the UCXX context calling
        `_deregister_dask_resource(resource_id)`, which stops the notifier thread
        and progress tasks when no more UCXX resources are alive.

        Returns
        -------
        resource_id: int
            The ID of the registered resource that should be used with
            `_deregister_dask_resource` during stop/destruction of the resource.
        """
        ctx = ucxx.core._get_ctx()
        with gc_disabled():
            # Disable garbage collection to avoid deadlocks should the garbage collector
            # run within the lock, where `_deregister_dask_resource` finalizer may be
            # called.
            with ctx._dask_resources_lock:
                resource_id = next(count)
                ctx._dask_resources.add(resource_id)
                ctx.start_notifier_thread()
                ctx.continuous_ucx_progress()
                return resource_id

    return register


_register_dask_resource = make_register()

del make_register


def _deregister_dask_resource(resource_id):
    """Deregister a Dask resource with the UCXX context.

    Deregister a Dask resource from the UCXX context with given ID, and if no
    resources remain after deregistration, stop the notifier thread and progress
    tasks.

    Parameters
    ----------
    resource_id: int
        The unique ID of the resource returned by `_register_dask_resource` upon
        registration.
    """
    if ucxx.core._ctx is None:
        # Prevent creation of context if it was already destroyed, all
        # registered references are already gone.
        return

    ctx = ucxx.core._get_ctx()

    # Check if the attribute exists first, in tests the UCXX context may have
    # been reset before some resources are deregistered.
    if hasattr(ctx, "_dask_resources_lock"):
        with ctx._dask_resources_lock:
            try:
                ctx._dask_resources.remove(resource_id)
            except KeyError:
                pass

            # Stop notifier thread and progress tasks if no Dask resources using
            # UCXX communicators are running anymore.
            if len(ctx._dask_resources) == 0:
                ucxx.stop_notifier_thread()
                ctx.clear_progress_tasks()


def _allocate_dask_resources_tracker() -> None:
    """Allocate Dask resources tracker.

    Allocate a Dask resources tracker in the UCXX context. This is useful to
    track Distributed communicators so that progress and notifier threads can
    be cleanly stopped when no UCXX communicators are alive anymore.
    """
    ctx = ucxx.core._get_ctx()
    if not hasattr(ctx, "_dask_resources"):
        # TODO: Move the `Lock` to a file/module-level variable for true
        # lock-safety. The approach implemented below could cause race
        # conditions if this function is called simultaneously by multiple
        # threads.
        from threading import Lock

        ctx._dask_resources = set()
        ctx._dask_resources_lock = Lock()


def init_once():
    global ucxx, device_array
    global ucx_create_endpoint, ucx_create_listener
    global pre_existing_cuda_context, cuda_context_created
    global multi_buffer

    if ucxx is not None:
        # Ensure reallocation of Dask resources tracker if the UCXX context was
        # reset since the previous `init_once()` call. This may happen in tests,
        # where the `ucxx_loop` fixture will reset the context after each test.
        _allocate_dask_resources_tracker()

        return

    # Set up configuration system with defaults
    setup_config()

    # remove/process dask.ucx flags for valid ucx options
    ucx_config, ucx_environment = _prepare_ucx_config()

    # We ensure the CUDA context is created before initializing UCX. This can't
    # be safely handled externally because communications in Dask start before
    # preload scripts run.
    # Precedence:
    # 1. external environment
    # 2. ucx_config (high level settings passed to ucxx.init)
    # 3. ucx_environment (low level settings equivalent to environment variables)
    ucx_tls = os.environ.get(
        "UCX_TLS",
        ucx_config.get("TLS", ucx_environment.get("UCX_TLS", "")),
    )
    if (
        get_ucx_config("create-cuda-context") is True
        # This is not foolproof, if UCX_TLS=all we might require CUDA
        # depending on configuration of UCX, but this is better than
        # nothing
        or ("cuda" in ucx_tls and "^cuda" not in ucx_tls)
    ):
        from ucxx._cuda_context import ensure_cuda_context

        cuda_visible_device = get_device_index_and_uuid(
            os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        )
        pre_existing_cuda_context = has_cuda_context()
        if pre_existing_cuda_context.has_context:
            _warn_existing_cuda_context(
                pre_existing_cuda_context.device_info, os.getpid()
            )

        ensure_cuda_context(0)

        cuda_context_created = has_cuda_context()
        if (
            cuda_context_created.has_context
            and cuda_context_created.device_info.uuid != cuda_visible_device.uuid
        ):
            _warn_cuda_context_wrong_device(
                cuda_visible_device, cuda_context_created.device_info, os.getpid()
            )

    multi_buffer = get_ucx_config("multi-buffer", default=False)

    import ucxx as _ucxx

    ucxx = _ucxx

    with patch.dict(os.environ, ucx_environment):
        # We carefully ensure that ucx_environment only contains things
        # that don't override ucx_config or existing slots in the
        # environment, so the user's external environment can safely
        # override things here.
        progress_mode = None if "UCXPY_PROGRESS_MODE" in os.environ else "blocking"
        ucxx.init(
            options=ucx_config, env_takes_precedence=True, progress_mode=progress_mode
        )
        _allocate_dask_resources_tracker()

    pool_size_str = get_rmm_config("pool-size")

    # Find the function, `cuda_array()`, to use when allocating new CUDA arrays.
    # RMM is required for CUDA array allocation at runtime (numba is only for tests).
    try:
        import rmm

        def device_array(n):
            return rmm.DeviceBuffer(size=n)

        if pool_size_str is not None:
            pool_size = parse_bytes(pool_size_str)
            rmm.reinitialize(
                pool_allocator=True, managed_memory=False, initial_pool_size=pool_size
            )
    except ImportError:

        def device_array(n):
            raise RuntimeError("In order to send/recv CUDA arrays, RMM is required.")

        if pool_size_str is not None:
            logger.warning(
                "Initial RMM pool size defined, but RMM is not available. "
                "Please consider installing RMM or removing the pool size option."
            )


def _close_comm(ref):
    """Callback to close Dask Comm when UCX Endpoint closes or errors

    Parameters
    ----------
        ref: weak reference to a Dask UCX comm
    """
    comm = ref()
    if comm is not None:
        comm._closed = True


def _finalizer(endpoint: ucxx.Endpoint, resource_id: int) -> None:
    """UCXX comms object finalizer.

    Attempt to close the UCXX endpoint if it's still alive, and deregister Dask
    resource.

    Parameters
    ----------
    endpoint: ucx_api.UCXEndpoint
        The endpoint to close.
    resource_id: int
        The unique ID of the resource returned by `_register_dask_resource` upon
        registration.
    """
    if endpoint is not None:
        endpoint.abort()
    _deregister_dask_resource(resource_id)


class UCXX(Comm):
    """Comm object using UCXX.

    Parameters
    ----------
    ep : ucxx.Endpoint
        The UCXX endpoint.
    local_address : str
        The local address, prefixed with `ucx://` to use.
    peer_address : str
        The address of the remote peer, prefixed with `ucx://` to use.
    deserialize : bool, default True
        Whether to deserialize data in :meth:`distributed.protocol.loads`
    enable_close_callback: : bool, default True
        Enable close callback, required so that the endpoint object is notified
        when the remote endpoint closed or errored. This is required for proper
        lifetime handling of UCX-Py, but should be disabled for UCXX.

    Notes
    -----
    The read-write cycle uses the following pattern:

    Each msg is serialized into a number of "data" frames. We prepend these
    real frames with two additional frames

        1. is_gpus: Boolean indicator for whether the frame should be
           received into GPU memory. Packed in '?' format. Unpack with
           ``<n_frames>?`` format.
        2. frame_size : Unsigned int describing the size of frame (in bytes)
           to receive. Packed in 'Q' format, so a length-0 frame is equivalent
           to an unsized frame. Unpacked with ``<n_frames>Q``.

    The expected read cycle is

    1. Read the frame describing if connection is closing and number of frames
    2. Read the frame describing whether each data frame is gpu-bound
    3. Read the frame describing whether each data frame is sized
    4. Read all the data frames.
    """

    def __init__(  # type: ignore[no-untyped-def]
        self,
        ep,
        local_addr: str,
        peer_addr: str,
        deserialize: bool = True,
        enable_close_callback: bool = False,
    ):
        super().__init__(deserialize=deserialize)
        self._ep = ep
        self._ep_handle = int(self._ep._ep.handle)
        self._write_lock = asyncio.Lock()
        self._closed = False
        if local_addr:
            assert local_addr.startswith(("ucx://", "ucxx://"))
        assert peer_addr.startswith(("ucx://", "ucxx://"))
        self._local_addr = local_addr
        self._peer_addr = peer_addr
        self.comm_flag = None

        if enable_close_callback:
            # When the UCX endpoint closes or errors the registered callback
            # is called.
            ref = weakref.ref(self)
            self._ep.set_close_callback(functools.partial(_close_comm, ref))
            self._has_close_callback = True
        else:
            self._has_close_callback = False

        resource_id = _register_dask_resource()
        self._resource_id = resource_id

        logger.debug("UCX.__init__ %s", self)

        weakref.finalize(self, _finalizer, ep, resource_id)

    @property
    def local_address(self) -> str:
        return self._local_addr

    @property
    def peer_address(self) -> str:
        return self._peer_addr

    @property
    def same_host(self) -> bool:
        """Unlike in TCP, local_address can be blank"""
        return super().same_host if self._local_addr else False

    @log_errors
    async def write(
        self,
        msg: dict,
        serializers: Collection[str] | None = None,
        on_error: str = "message",
    ) -> int:
        async with self._write_lock:
            if self._closed or self.closed():
                raise CommClosedError("Endpoint is closed -- unable to send message")
            return await self._write_unlocked(msg, serializers, on_error)

    async def _write_unlocked(
        self,
        msg: dict,
        serializers: Collection[str] | None,
        on_error: str,
    ) -> int:
        if serializers is None:
            serializers = ("cuda", "dask", "pickle", "error")
        # msg can also be a list of dicts when sending batched messages
        frames = await to_frames(
            msg,
            serializers=serializers,
            on_error=on_error,
            allow_offload=self.allow_offload,
        )
        sizes = tuple(nbytes(f) for f in frames)

        try:
            if multi_buffer is True:
                if any(hasattr(f, "__cuda_array_interface__") for f in frames):
                    synchronize_stream(CudaStream.Default)

                close = [struct.pack("?", False)]
                await self.ep.send_multi(close + frames)
            else:
                nframes = len(frames)
                cuda_frames = tuple(
                    hasattr(f, "__cuda_array_interface__") for f in frames
                )
                cuda_send_frames, send_frames = zip(
                    *(
                        (is_cuda, each_frame)
                        for is_cuda, each_frame in zip(cuda_frames, frames)
                        if nbytes(each_frame) > 0
                    )
                )

                # Send meta data

                # Send close flag and number of frames (_Bool, int64)
                await self.ep.send(struct.pack("?Q", False, nframes))
                # Send which frames are CUDA (bool) and
                # how large each frame is (uint64)
                await self.ep.send(
                    struct.pack(nframes * "?" + nframes * "Q", *cuda_frames, *sizes)
                )

                # Send frames

                # It is necessary to first synchronize the default stream before start
                # sending We synchronize the default stream because UCX is not
                # stream-ordered and syncing the default stream will wait for other
                # non-blocking CUDA streams. Note this is only sufficient if the memory
                # being sent is not currently in use on non-blocking CUDA streams.
                if any(cuda_send_frames):
                    synchronize_stream(CudaStream.Default)

                for each_frame in send_frames:
                    await self.ep.send(each_frame)
            return sum(sizes)
        except ucxx.exceptions.UCXError:
            self.abort()
            raise CommClosedError("While writing, the connection was closed")

    @log_errors
    async def read(self, deserializers=("cuda", "dask", "pickle", "error")):
        if deserializers is None:
            deserializers = ("cuda", "dask", "pickle", "error")

        if multi_buffer is True:
            try:
                # TODO: We don't know if any frames are CUDA, investigate whether
                # we need to synchronize device here.
                frames = await self.ep.recv_multi()
                shutdown_frame = frames[0]
                frames = frames[1:]

                (shutdown,) = struct.unpack("?", shutdown_frame)

                if shutdown:  # The writer is closing the connection
                    raise CommClosedError("Connection closed by writer")
            except BaseException as e:
                # In addition to UCX exceptions, may be CancelledError or another
                # "low-level" exception. The only safe thing to do is to abort.
                # (See also https://github.com/dask/distributed/pull/6574).
                self.abort()
                raise CommClosedError("Connection closed by writer") from e
        else:
            try:
                # Recv meta data

                # Recv close flag and number of frames (_Bool, int64)
                msg = host_array(struct.calcsize("?Q"))
                await self.ep.recv(msg)
                (shutdown, nframes) = struct.unpack("?Q", msg)

                if shutdown:  # The writer is closing the connection
                    raise CommClosedError("Connection closed by writer")

                # Recv which frames are CUDA (bool) and
                # how large each frame is (uint64)
                header_fmt = nframes * "?" + nframes * "Q"
                header = host_array(struct.calcsize(header_fmt))
                await self.ep.recv(header)
                header = struct.unpack(header_fmt, header)
                cuda_frames, sizes = header[:nframes], header[nframes:]
            except BaseException as e:
                # In addition to UCX exceptions, may be CancelledError or another
                # "low-level" exception. The only safe thing to do is to abort.
                # (See also https://github.com/dask/distributed/pull/6574).
                self.abort()
                raise CommClosedError("Connection closed by writer") from e
            else:
                try:
                    # Recv frames
                    frames = [
                        device_array(each_size) if is_cuda else host_array(each_size)
                        for is_cuda, each_size in zip(cuda_frames, sizes)
                    ]
                    cuda_recv_frames, recv_frames = zip(
                        *(
                            (is_cuda, each_frame)
                            for is_cuda, each_frame in zip(cuda_frames, frames)
                            if nbytes(each_frame) > 0
                        )
                    )

                    # It is necessary to first populate `frames` with CUDA arrays
                    # and synchronize the default stream before starting receiving
                    # to ensure buffers have been allocated
                    if any(cuda_recv_frames):
                        synchronize_stream(CudaStream.Default)

                    for each_frame in recv_frames:
                        await self.ep.recv(each_frame)
                except BaseException as e:
                    # In addition to UCX exceptions, may be CancelledError or another
                    # "low-level" exception. The only safe thing to do is to abort.
                    # (See also https://github.com/dask/distributed/pull/6574).
                    self.abort()
                    raise CommClosedError("Connection closed by writer") from e

        try:
            return await from_frames(
                frames,
                deserialize=self.deserialize,
                deserializers=deserializers,
                allow_offload=self.allow_offload,
            )
        except EOFError:
            # Frames possibly garbled or truncated by communication error
            self.abort()
            raise CommClosedError("Aborted stream on truncated data")

    async def close(self):
        async with self._write_lock:
            self._closed = True
            if self._ep is not None:
                try:
                    if multi_buffer is True:
                        await self.ep.send_multi([struct.pack("?", True)])
                    else:
                        await self.ep.send(struct.pack("?Q", True, 0))
                except (
                    ucxx.exceptions.UCXError,
                    ucxx.exceptions.UCXCloseError,
                    ucxx.exceptions.UCXCanceledError,
                    ucxx.exceptions.UCXConnectionResetError,
                    ucxx.exceptions.UCXUnreachableError,
                ):
                    # If the other end is in the process of closing,
                    # UCX will sometimes raise a `Input/output` error,
                    # which we can ignore.
                    pass
                self.abort()
                self._ep = None

    def abort(self):
        self._closed = True
        if self._ep is not None:
            self._ep.abort()
            self._ep = None
            _deregister_dask_resource(self._resource_id)

    def closed(self):
        if self._has_close_callback is True:
            return self._closed
        if self._ep is not None:
            # Even if the endpoint has not been closed or aborted by Dask, the lifetime
            # of the underlying endpoint may have changed if the remote endpoint has
            # disconnected. In that case there may still exist enqueued messages on its
            # buffer to be received, even though sending is not possible anymore.
            return not self._ep.alive
        else:
            return True

    @property
    def ep(self):
        if self._ep is not None:
            return self._ep
        else:
            raise CommClosedError("UCX Endpoint is closed")


class UCXXConnector(Connector):
    prefix = "ucx://"
    comm_class = UCXX
    encrypted = False

    async def connect(
        self, address: str, deserialize: bool = True, **connection_args: Any
    ) -> UCXX:
        logger.debug("UCXXConnector.connect: %s", address)
        ip, port = parse_host_port(address)
        init_once()

        try:
            self._resource_id = _register_dask_resource()
            ep = await ucxx.create_endpoint(ip, port)
        except (
            ucxx.exceptions.UCXCloseError,
            ucxx.exceptions.UCXCanceledError,
            ucxx.exceptions.UCXConnectionResetError,
            ucxx.exceptions.UCXMessageTruncatedError,
            ucxx.exceptions.UCXNotConnectedError,
            ucxx.exceptions.UCXUnreachableError,
        ):
            raise CommClosedError("Connection closed before handshake completed")
        finally:
            _deregister_dask_resource(self._resource_id)
        return self.comm_class(
            ep,
            local_addr="",
            peer_addr=self.prefix + address,
            deserialize=deserialize,
        )


class UCXXListener(Listener):
    prefix = UCXXConnector.prefix
    comm_class = UCXXConnector.comm_class
    encrypted = UCXXConnector.encrypted

    def __init__(
        self,
        address: str,
        comm_handler: Callable[[UCXX], Awaitable[None]] | None = None,
        deserialize: bool = False,
        allow_offload: bool = True,
        **connection_args: Any,
    ):
        if not address.startswith(self.prefix):
            address = f"{self.prefix}{address}"
        self.ip, self._input_port = parse_host_port(address, default_port=0)
        self.comm_handler = comm_handler
        self.deserialize = deserialize
        self.allow_offload = allow_offload
        self._ep = None  # type: ucxx.Endpoint
        self.ucxx_server = None
        self.connection_args = connection_args

    @property
    def port(self):
        return self.ucxx_server.port

    @property
    def address(self):
        return f"{self.prefix}{self.ip}:{self.port}"

    async def start(self):
        async def serve_forever(client_ep):
            ucx = self.comm_class(
                client_ep,
                local_addr=self.address,
                peer_addr=self.address,
                deserialize=self.deserialize,
            )
            ucx.allow_offload = self.allow_offload
            try:
                await self.on_connection(ucx)
            except CommClosedError:
                logger.debug("Connection closed before handshake completed")
                return
            if self.comm_handler:
                await self.comm_handler(ucx)

        init_once()
        self._resource_id = _register_dask_resource()
        weakref.finalize(self, _deregister_dask_resource, self._resource_id)
        self.ucxx_server = ucxx.create_listener(serve_forever, port=self._input_port)

    def stop(self):
        self.ucxx_server = None
        _deregister_dask_resource(self._resource_id)

    def get_host_port(self):
        # TODO: TCP raises if this hasn't started yet.
        return self.ip, self.port

    @property
    def listen_address(self):
        return self.prefix + unparse_host_port(*self.get_host_port())

    @property
    def contact_address(self):
        host, port = self.get_host_port()
        host = ensure_concrete_host(host)  # TODO: ensure_concrete_host
        return self.prefix + unparse_host_port(host, port)

    @property
    def bound_address(self):
        # TODO: Does this become part of the base API? Kinda hazy, since
        # we exclude in for inproc.
        return self.get_host_port()


class UCXXBackend(Backend):
    # I / O

    def get_connector(self):
        return UCXXConnector()

    def get_listener(self, loc, handle_comm, deserialize, **connection_args):
        return UCXXListener(loc, handle_comm, deserialize, **connection_args)

    # Address handling
    # This duplicates BaseTCPBackend

    def get_address_host(self, loc):
        return parse_host_port(loc)[0]

    def get_address_host_port(self, loc):
        return parse_host_port(loc)

    def resolve_address(self, loc):
        host, port = parse_host_port(loc)
        return unparse_host_port(ensure_ip(host), port)

    def get_local_address_for(self, loc):
        host, port = parse_host_port(loc)
        host = ensure_ip(host)
        if ":" in host:
            local_host = get_ipv6(host)
        else:
            local_host = get_ip(host)
        return unparse_host_port(local_host, None)


def _prepare_ucx_config():
    """Translate dask config options to appropriate UCX config options

    Returns
    -------
    tuple
        Options suitable for passing to ``ucxxp.init`` and additional
        UCX options that will be inserted directly into the environment
        while calling ``ucxx.init``.
    """

    # configuration of UCX can happen in two ways:
    # 1) high level on/off flags which correspond to UCX configuration
    # 2) explicitly defined UCX configuration flags in distributed.comm.ucx.environment
    # High-level settings in (1) are preferred to settings in (2)
    # Settings in the external environment override both

    high_level_options = {}

    # if any of the high level flags are set, as long as they are not Null/None,
    # we assume we should configure basic TLS settings for UCX, otherwise we
    # leave UCX to its default configuration
    if any(
        [
            get_ucx_config("tcp"),
            get_ucx_config("nvlink"),
            get_ucx_config("infiniband"),
        ]
    ):
        if get_ucx_config("rdmacm"):
            tls = "tcp"
            tls_priority = "rdmacm"
        else:
            tls = "tcp"
            tls_priority = "tcp"

        # CUDA COPY can optionally be used with ucx -- we rely on the user
        # to define when messages will include CUDA objects.  Note:
        # defining only the Infiniband flag will not enable cuda_copy
        if any(
            [
                get_ucx_config("nvlink"),
                get_ucx_config("cuda-copy"),
            ]
        ):
            tls = tls + ",cuda_copy"

        if get_ucx_config("infiniband"):
            tls = "rc," + tls
        if get_ucx_config("nvlink"):
            tls = tls + ",cuda_ipc"

        high_level_options = {"TLS": tls, "SOCKADDR_TLS_PRIORITY": tls_priority}

    # Pick up any other ucx environment settings
    environment_options = {}
    for k, v in get_ucx_config("environment", default={}).items():
        # {"some-name": value} is translated to {"UCX_SOME_NAME": value}
        key = "_".join(map(str.upper, ("UCX", *k.split("-"))))
        if (hl_key := key[4:]) in high_level_options:
            logger.warning(
                f"Ignoring {k}={v} ({key=}) in ucx.environment, "
                f"preferring {hl_key}={high_level_options[hl_key]} "
                "from high level options"
            )
        elif key in os.environ:
            # This is only info because setting UCX configuration via
            # environment variables is a reasonably common approach
            logger.info(
                f"Ignoring {k}={v} ({key=}) in ucx.environment, "
                f"preferring {key}={os.environ[key]} from external environment"
            )
        else:
            environment_options[key] = v

    return high_level_options, environment_options


class UCXXLegacyPrefix(UCXX):
    prefix = "ucxx://"


class UCXXConnectorLegacyPrefix(UCXXConnector):
    prefix = "ucxx://"
    comm_class = UCXXLegacyPrefix


class UCXXListenerLegacyPrefix(UCXXListener):
    prefix = UCXXConnectorLegacyPrefix.prefix
    comm_class = UCXXConnectorLegacyPrefix.comm_class
    encrypted = UCXXConnectorLegacyPrefix.encrypted


class UCXXBackendLegacyPrefix(UCXXBackend):
    def get_connector(self):
        return UCXXConnectorLegacyPrefix()

    def get_listener(self, loc, handle_comm, deserialize, **connection_args):
        return UCXXListenerLegacyPrefix(
            loc, handle_comm, deserialize, **connection_args
        )
