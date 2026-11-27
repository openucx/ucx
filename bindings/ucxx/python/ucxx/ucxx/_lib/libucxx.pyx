# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


import asyncio
import enum
import functools
from pickle import PickleBuffer
import logging
import weakref
from typing import Optional

from cpython.buffer cimport PyBuffer_FillInfo
from cpython.ref cimport PyObject
from cython.operator cimport dereference as deref
from libc.stdint cimport uint8_t, uintptr_t
from libc.stdlib cimport free
from libcpp cimport nullptr
from libcpp.functional cimport function
from libcpp.memory cimport (
    dynamic_pointer_cast,
    make_shared,
    make_unique,
    shared_ptr,
    static_pointer_cast,
)
from libcpp.string cimport string
from libcpp.string_view cimport string_view
from libcpp.utility cimport move
from libcpp.vector cimport vector

cdef extern from "cuda_runtime.h" nogil:
    enum cudaMemcpyKind:
        cudaMemcpyDeviceToHost
    int cudaMemcpy(void* dst, const void* src, size_t count,
                   cudaMemcpyKind kind)

import numpy as np

from .arr cimport Array
from .ucxx_api cimport *

include "tag.pyx"

logger = logging.getLogger("ucx")


cdef class TagProbeResult:
    """Result of a tag probe operation.

    This class provides a clean, typed interface for accessing tag probe results.
    It wraps the C++ TagProbeInfo structure and provides Python-friendly access
    to the probe information.
    """

    cdef shared_ptr[TagProbeInfo] _probe_info_ptr

    @property
    def matched(self) -> bool:
        """Whether a message was matched.

        Returns:
            True if a message was matched, False otherwise.
        """
        return self._probe_info_ptr.get().isMatched()

    @property
    def sender_tag(self) -> UCXXTag:
        """The sender tag of the matched message.

        Returns:
            The sender tag value.

        Raises:
            AttributeError: If no message was matched.
        """
        if not self.matched:
            raise AttributeError("No message was matched")
        return UCXXTag(self._probe_info_ptr.get().getInfo().senderTag)

    @property
    def length(self) -> int:
        """The length of the matched message.

        Returns:
            The message length in bytes.

        Raises:
            AttributeError: If no message was matched.
        """
        if not self.matched:
            raise AttributeError("No message was matched")
        return self._probe_info_ptr.get().getInfo().length

    @property
    def handle(self) -> Optional[int]:
        """The message handle for efficient reception.

        Returns:
            The message handle as an integer, or None if no handle is available.

        Raises:
            AttributeError: If no message was matched.
        """
        if not self.matched:
            raise AttributeError("No message was matched")

        cdef ucp_tag_message_h handle
        cdef uintptr_t handle_ptr

        try:
            handle = self._probe_info_ptr.get().getHandle()
            if handle == NULL:
                return None
            handle_ptr = <uintptr_t>handle
        except RuntimeError:
            return None
        return int(handle_ptr)

    def __bool__(self) -> bool:
        """Return whether a message was matched.

        Returns:
            True if a message was matched, False otherwise.
        """
        return self.matched

    def __repr__(self) -> str:
        """String representation of the probe result.

        Returns:
            A string representation showing the probe result state.
        """
        if self.matched:
            handle_val = self.handle
            result = (
                "TagProbeResult("
                f"matched=True, sender_tag={self.sender_tag}, length={self.length}"
                + (f", handle={handle_val}" if handle_val is not None else "")
                + ")"
            )
            return result
        else:
            return "TagProbeResult(matched=False)"


cdef class HostBufferAdapter:
    """A simple adapter around HostBuffer implementing the buffer protocol"""

    @staticmethod
    cdef _from_host_buffer(HostBuffer* host_buffer):
        """Construct a new HostBufferAdapter from a HostBuffer.

        This factory takes ownership of the input host_buffer's data, so
        attempting to use the input after this function is called will result
        in undefined behavior.
        """
        cdef HostBufferAdapter obj = HostBufferAdapter.__new__(HostBufferAdapter)
        obj._size = host_buffer.getSize()
        obj._ptr = host_buffer.release()
        obj._shape = [obj._size]
        obj._itemsize = sizeof(uint8_t)
        obj._strides = [obj._itemsize]
        return obj

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        buffer.buf = self._ptr
        buffer.format = 'B'
        buffer.internal = NULL
        buffer.itemsize = self._itemsize
        buffer.len = self._size * self._itemsize
        buffer.ndim = 1
        buffer.readonly = 0
        buffer.shape = self._shape
        buffer.strides = self._strides
        buffer.suboffsets = NULL
        buffer.obj = self

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __dealloc__(self):
        free(self._ptr)


def _get_host_buffer(uintptr_t recv_buffer_ptr):
    cdef HostBuffer* host_buffer = <HostBuffer*>recv_buffer_ptr
    return np.asarray(HostBufferAdapter._from_host_buffer(host_buffer))


# CCCL has no Python buffer equivalent. This wrapper provides __cuda_array_interface__
# for interoperability with CuPy/cuDF without requiring an external CCCL Python package.
cdef class CCCLBufferWrapper:
    """Wraps a CCCLBuffer device pointer as cuda_array_interface object.

    Retains a shared_ptr to the underlying Buffer to prevent the CUDA
    allocation from being freed while this wrapper is alive.
    """
    cdef shared_ptr[Buffer] _buf
    cdef uintptr_t _ptr
    cdef size_t _size

    @staticmethod
    cdef CCCLBufferWrapper _from_buffer(shared_ptr[Buffer] buf):
        cdef CCCLBufferWrapper wrapper = CCCLBufferWrapper.__new__(CCCLBufferWrapper)
        wrapper._buf = buf
        wrapper._ptr = <uintptr_t>buf.get().data()
        wrapper._size = buf.get().getSize()
        return wrapper

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self._size,),
            "typestr": "|u1",
            "data": (self._ptr, False),
            "version": 2,
        }

    @property
    def nbytes(self):
        return self._size

    def copy_to_host(self, ary=None):
        """Copy device buffer content to host.

        Parameters
        ----------
        ary : optional
            A bytes-like buffer to write into. If None, a new numpy
            array is allocated.

        Returns
        -------
        numpy.ndarray
            Host array with dtype uint8.
        """
        cdef size_t s = self._size
        cdef unsigned char[::1] hb
        if ary is None:
            ary = np.empty((s,), dtype="u1")
        hb = ary
        if <size_t>len(hb) < s:
            raise ValueError(
                f"Argument `ary` is too small. Need space for {s} bytes."
            )
        with nogil:
            cudaMemcpy(<void*>&hb[0], <const void*>self._ptr, s,
                       cudaMemcpyDeviceToHost)
        return ary

cdef _get_cccl_buffer(shared_ptr[Buffer] buf):
    return CCCLBufferWrapper._from_buffer(buf)


cdef shared_ptr[Buffer] _cccl_am_allocator(size_t length) noexcept nogil:
    cdef shared_ptr[CCCLBuffer] cccl_buffer = make_shared[CCCLBuffer](length)
    return dynamic_pointer_cast[Buffer, CCCLBuffer](cccl_buffer)

###############################################################################
#                               Exceptions                                    #
###############################################################################

UCXError = None

UCXNoMessageError = None
UCXNoResourceError = None
UCXIOError = None
UCXNoMemoryError = None
UCXInvalidParamError = None
UCXUnreachableError = None
UCXInvalidAddrError = None
UCXNotImplementedError = None
UCXMessageTruncatedError = None
UCXNoProgressError = None
UCXBufferTooSmallError = None
UCXNoElemError = None
UCXSomeConnectsFailedError = None
UCXNoDeviceError = None
UCXBusyError = None
UCXCanceledError = None
UCXShmemSegmentError = None
UCXAlreadyExistsError = None
UCXOutOfRangeError = None
UCXTimedOutError = None
UCXExceedsLimitError = None
UCXUnsupportedError = None
UCXRejectedError = None
UCXNotConnectedError = None
UCXConnectionResetError = None
UCXFirstLinkFailureError = None
UCXLastLinkFailureError = None
UCXFirstEndpointFailureError = None
UCXEndpointTimeoutError = None
UCXLastEndpointFailureError = None

UCXCloseError = None
UCXConfigError = None

# Legacy names
UCXCanceled = None
UCXMsgTruncated = None


def _create_exceptions():
    global UCXError

    global UCXNoMessageError
    global UCXNoResourceError
    global UCXIOError
    global UCXNoMemoryError
    global UCXInvalidParamError
    global UCXUnreachableError
    global UCXInvalidAddrError
    global UCXNotImplementedError
    global UCXMessageTruncatedError
    global UCXNoProgressError
    global UCXBufferTooSmallError
    global UCXNoElemError
    global UCXSomeConnectsFailedError
    global UCXNoDeviceError
    global UCXBusyError
    global UCXCanceledError
    global UCXShmemSegmentError
    global UCXAlreadyExistsError
    global UCXOutOfRangeError
    global UCXTimedOutError
    global UCXExceedsLimitError
    global UCXUnsupportedError
    global UCXRejectedError
    global UCXNotConnectedError
    global UCXConnectionResetError
    global UCXFirstLinkFailureError
    global UCXLastLinkFailureError
    global UCXFirstEndpointFailureError
    global UCXEndpointTimeoutError
    global UCXLastEndpointFailureError

    global UCXCloseError
    global UCXConfigError

    # Legacy names
    global UCXCanceled
    global UCXMsgTruncated

    create_exceptions()

    UCXError = <object>UCXXError

    UCXNoMessageError = <object>UCXXNoMessageError
    UCXNoResourceError = <object>UCXXNoResourceError
    UCXIOError = <object>UCXXIOError
    UCXNoMemoryError = <object>UCXXNoMemoryError
    UCXInvalidParamError = <object>UCXXInvalidParamError
    UCXUnreachableError = <object>UCXXUnreachableError
    UCXInvalidAddrError = <object>UCXXInvalidAddrError
    UCXNotImplementedError = <object>UCXXNotImplementedError
    UCXMessageTruncatedError = <object>UCXXMessageTruncatedError
    UCXNoProgressError = <object>UCXXNoProgressError
    UCXBufferTooSmallError = <object>UCXXBufferTooSmallError
    UCXNoElemError = <object>UCXXNoElemError
    UCXSomeConnectsFailedError = <object>UCXXSomeConnectsFailedError
    UCXNoDeviceError = <object>UCXXNoDeviceError
    UCXBusyError = <object>UCXXBusyError
    UCXCanceledError = <object>UCXXCanceledError
    UCXShmemSegmentError = <object>UCXXShmemSegmentError
    UCXAlreadyExistsError = <object>UCXXAlreadyExistsError
    UCXOutOfRangeError = <object>UCXXOutOfRangeError
    UCXTimedOutError = <object>UCXXTimedOutError
    UCXExceedsLimitError = <object>UCXXExceedsLimitError
    UCXUnsupportedError = <object>UCXXUnsupportedError
    UCXRejectedError = <object>UCXXRejectedError
    UCXNotConnectedError = <object>UCXXNotConnectedError
    UCXConnectionResetError = <object>UCXXConnectionResetError
    UCXFirstLinkFailureError = <object>UCXXFirstLinkFailureError
    UCXLastLinkFailureError = <object>UCXXLastLinkFailureError
    UCXFirstEndpointFailureError = <object>UCXXFirstEndpointFailureError
    UCXEndpointTimeoutError = <object>UCXXEndpointTimeoutError
    UCXLastEndpointFailureError = <object>UCXXLastEndpointFailureError

    UCXCloseError = <object>UCXXCloseError
    UCXConfigError = <object>UCXXConfigError

    # Define legacy names
    # TODO: Deprecate and remove
    UCXCanceled = UCXCanceledError
    UCXMsgTruncated = UCXMessageTruncatedError


###############################################################################
#                                   Types                                     #
###############################################################################

class Feature(enum.Enum):
    TAG = UCP_FEATURE_TAG
    RMA = UCP_FEATURE_RMA
    AMO32 = UCP_FEATURE_AMO32
    AMO64 = UCP_FEATURE_AMO64
    WAKEUP = UCP_FEATURE_WAKEUP
    STREAM = UCP_FEATURE_STREAM
    AM = UCP_FEATURE_AM


class PythonRequestNotifierWaitState(enum.Enum):
    Ready = RequestNotifierWaitState.Ready
    Timeout = RequestNotifierWaitState.Timeout
    Shutdown = RequestNotifierWaitState.Shutdown


###############################################################################
#                                   Classes                                   #
###############################################################################

cdef class UCXConfig():
    def __init__(self, ConfigMap user_options=ConfigMap()) -> None:
        # TODO: Replace unique_ptr by stack object. Rule-of-five is not allowed
        # by Config, and Cython seems not to handle constructors without moving
        # in `__init__`.
        self._config = move(make_unique[Config](user_options))

    def __dealloc__(self) -> None:
        with nogil:
            self._config.reset()

    @property
    def config(self) -> dict:
        cdef ConfigMap config_map = self._config.get().get()
        return {
            item.first.decode("utf-8"): item.second.decode("utf-8")
            for item in config_map
        }


cdef class UCXContext():
    """Python representation of `ucp_context_h`

    Parameters
    ----------
    config_dict: Mapping[str, str]
        UCX options such as "MEMTYPE_CACHE=n" and "SEG_SIZE=3M"
    feature_flags: Iterable[Feature]
        Tuple of UCX feature flags
    """
    def __init__(
        self,
        dict config_dict=None,
        tuple feature_flags=(
            Feature.TAG,
            Feature.WAKEUP,
            Feature.STREAM,
            Feature.AM,
            Feature.RMA
        )
    ) -> None:
        cdef unique_ptr[ContextBuilder] builder
        cdef ConfigMap cpp_config_in, cpp_config_out
        cdef dict context_config

        if config_dict is None:
            config_dict = {}

        for k, v in config_dict.items():
            cpp_config_in[k.encode("utf-8")] = v.encode("utf-8")
        cdef uint64_t feature_flags_uint = functools.reduce(
            lambda x, y: x | y.value, feature_flags, 0
        )

        with nogil:
            # Use unique_ptr for heap allocation to avoid requiring a default
            # constructor, Cython requires stack-allocated C++ objects to have nullary
            # constructors
            builder = make_unique[ContextBuilder](feature_flags_uint)
            builder.get().configMap(cpp_config_in)
            self._context = builder.get().build()
            cpp_config_out = self._context.get().getConfig()

        context_config = cpp_config_out

        self._config = {
            k.decode("utf-8"): v.decode("utf-8") for k, v in context_config.items()
        }

        logger.info("UCP initiated using config: ")
        for k, v in self._config.items():
            logger.info(f"  {k}, {v}")

    def __dealloc__(self) -> None:
        with nogil:
            self._context.reset()

    @property
    def config(self) -> dict:
        return self._config

    @property
    def feature_flags(self) -> int:
        return int(self._context.get().getFeatureFlags())

    @property
    def cuda_support(self) -> bool:
        return bool(self._context.get().hasCudaSupport())

    @property
    def handle(self) -> int:
        cdef ucp_context_h handle

        with nogil:
            handle = self._context.get().getHandle()

        return int(<uintptr_t>handle)

    @property
    def ucxx_ptr(self) -> int:
        cdef Context* context

        with nogil:
            context = self._context.get()

        return int(<uintptr_t>context)

    cdef shared_ptr[Context] get_ucxx_shared_ptr(self) nogil:
        return self._context

    @property
    def info(self) -> str:
        cdef Context* ucxx_context
        cdef string info

        with nogil:
            ucxx_context = self._context.get()
            info = ucxx_context.getInfo()

        return info.decode("utf-8")


cdef class UCXAddress():
    def __init__(self) -> None:
        raise TypeError("UCXAddress cannot be instantiated directly.")

    def __dealloc__(self) -> None:
        with nogil:
            self._handle = NULL
            self._address.reset()

    @classmethod
    def create_from_worker(cls, UCXWorker worker) -> UCXAddress:
        cdef UCXAddress address = UCXAddress.__new__(UCXAddress)
        cdef unique_ptr[AddressBuilder] builder

        address._bytes = None
        with nogil:
            builder = make_unique[AddressBuilder](worker._worker)
            address._address = builder.get().build()
            address._handle = address._address.get().getHandle()
            address._length = address._address.get().getLength()
            address._string = address._address.get().getStringView()

        return address

    @classmethod
    def create_from_buffer(cls, const uint8_t[::1] buf) -> UCXAddress:
        cdef UCXAddress address = UCXAddress.__new__(UCXAddress)
        cdef string_view address_strv = string_view(<const char*>&buf[0], len(buf))
        cdef unique_ptr[AddressBuilder] builder

        address._bytes = None
        with nogil:
            builder = make_unique[AddressBuilder](address_strv)
            address._address = builder.get().build()
            address._handle = address._address.get().getHandle()
            address._length = address._address.get().getLength()
            address._string = address._address.get().getStringView()

        return address

    @property
    def address(self) -> int:
        return int(<uintptr_t>self._handle)

    @property
    def ucxx_ptr(self) -> int:
        cdef Address* address

        with nogil:
            address = self._address.get()

        return int(<uintptr_t>address)

    cdef shared_ptr[Address] get_ucxx_shared_ptr(self) nogil:
        return self._address

    cpdef bytes tobytes(self):
        if self._bytes is None:
            self._bytes = bytes(self._string)
        return self._bytes

    @property
    def length(self) -> int:
        return int(self._length)

    def __bytes__(self) -> bytes:
        return self.tobytes()

    def __getbuffer__(self, Py_buffer *buffer, int flags) -> None:
        PyBuffer_FillInfo(buffer, self, self._handle, self._length, True, flags)

    def __releasebuffer__(self, Py_buffer *buffer) -> None:
        pass

    def __reduce_ex__(self, Py_ssize_t protocol) -> tuple:
        if protocol >= 5:
            return (UCXAddress.create_from_buffer, (PickleBuffer(self),))
        else:
            return (UCXAddress.create_from_buffer, (self.tobytes(),))

    def __hash__(self) -> Py_hash_t:
        return hash(self.tobytes())


cdef void _generic_callback(void *args) with gil:
    """Callback function called when UCXEndpoint closes or errors"""
    cdef dict cb_data = <dict> args

    try:
        cb_data['cb_func'](
            *cb_data['cb_args'],
            **cb_data['cb_kwargs']
        )
    except Exception as e:
        pass


cdef class UCXWorker():
    """Python representation of `ucp_worker_h`"""
    def __init__(
            self,
            UCXContext context,
            bint enable_delayed_submission=False,
            bint enable_python_future=False,
    ) -> None:
        cdef bint ucxx_enable_delayed_submission = enable_delayed_submission
        cdef bint ucxx_enable_python_future = enable_python_future
        cdef AmAllocatorType cccl_am_allocator

        self._context_feature_flags = <uint64_t>(context.feature_flags)

        with nogil:
            self._worker = createPythonWorker(
                context._context,
                ucxx_enable_delayed_submission,
                ucxx_enable_python_future,
            )
            self._enable_delayed_submission = (
                self._worker.get().isDelayedRequestSubmissionEnabled()
            )
            self._enable_python_future = self._worker.get().isFutureEnabled()

            if self._context_feature_flags & UCP_FEATURE_AM:
                if self._worker.get().getCudaBufferType() == BufferType.CCCL:
                    cccl_am_allocator = <AmAllocatorType>(&_cccl_am_allocator)
                    self._worker.get().registerAmAllocator(
                        UCS_MEMORY_TYPE_CUDA, cccl_am_allocator
                    )

    def __dealloc__(self) -> None:
        with nogil:
            self._worker.reset()

    @property
    def handle(self) -> int:
        cdef ucp_worker_h handle

        with nogil:
            handle = self._worker.get().getHandle()

        return int(<uintptr_t>handle)

    @property
    def ucxx_ptr(self) -> int:
        cdef Worker* worker

        with nogil:
            worker = self._worker.get()

        return int(<uintptr_t>worker)

    cdef shared_ptr[Worker] get_ucxx_shared_ptr(self) nogil:
        return self._worker

    @property
    def info(self) -> str:
        cdef Worker* ucxx_worker
        cdef string info

        with nogil:
            ucxx_worker = self._worker.get()
            info = ucxx_worker.getInfo()

        return info.decode("utf-8")

    @property
    def address(self) -> UCXAddress:
        return UCXAddress.create_from_worker(self)

    @property
    def enable_delayed_submission(self) -> bool:
        return self._enable_delayed_submission

    @property
    def enable_python_future(self) -> bool:
        return self._enable_python_future

    @property
    def cuda_buffer_type(self) -> str:
        """Return the preferred CUDA buffer type ('cccl', or 'none')."""
        cdef BufferType bt
        with nogil:
            bt = self._worker.get().getCudaBufferType()
        if bt == BufferType.CCCL:
            return "cccl"
        else:
            return "none"

    def create_endpoint_from_hostname(
            self,
            str ip_address,
            uint16_t port,
            bint endpoint_error_handling
    ) -> UCXEndpoint:
        return UCXEndpoint.create(self, ip_address, port, endpoint_error_handling)

    def create_endpoint_from_worker_address(
            self,
            UCXAddress address,
            bint endpoint_error_handling
    ) -> UCXEndpoint:
        return UCXEndpoint.create_from_worker_address(
            self, address, endpoint_error_handling
        )

    def init_blocking_progress_mode(self) -> None:
        with nogil:
            self._worker.get().initBlockingProgressMode()

    def arm(self) -> bool:
        cdef bint armed

        with nogil:
            armed = self._worker.get().arm()

        return armed

    @property
    def epoll_file_descriptor(self) -> int:
        cdef int epoll_file_descriptor = 0

        with nogil:
            epoll_file_descriptor = self._worker.get().getEpollFileDescriptor()

        return epoll_file_descriptor

    def progress(self) -> None:
        with nogil:
            self._worker.get().progress()

    def progress_once(self) -> bool:
        cdef bint progress_made

        with nogil:
            progress_made = self._worker.get().progressOnce()

        return progress_made

    def progress_worker_event(self, int epoll_timeout=-1) -> None:
        cdef int ucxx_epoll_timeout = epoll_timeout

        with nogil:
            self._worker.get().progressWorkerEvent(ucxx_epoll_timeout)

    def start_progress_thread(
        self,
        bint polling_mode=False,
        int epoll_timeout=-1
    ) -> None:
        cdef int ucxx_epoll_timeout = epoll_timeout

        with nogil:
            self._worker.get().startProgressThread(
                polling_mode, epoll_timeout=ucxx_epoll_timeout
            )

    def stop_progress_thread(self) -> None:
        with nogil:
            self._worker.get().stopProgressThread()

    def cancel_inflight_requests(
        self,
        uint64_t period=0,
        uint64_t max_attempts=1
    ) -> int:
        cdef uint64_t c_period = period
        cdef uint64_t c_max_attempts = max_attempts
        cdef size_t num_canceled

        with nogil:
            num_canceled = self._worker.get().cancelInflightRequests(
                c_period, c_max_attempts
            )

        return num_canceled

    def tag_probe(
        self,
        UCXXTag tag,
        UCXXTagMask tag_mask = UCXXTagMaskFull,
        bint remove = False
    ) -> TagProbeResult:
        """Probe for tag messages.

        Returns
        -------
        TagProbeResult object containing:
        - matched: bool indicating if a message was matched
        - sender_tag: int sender tag (when matched=True)
        - length: int message length in bytes (when matched=True)
        - handle: int message handle for efficient reception (when matched=True and
                  remove=True)
        """
        cdef Tag cpp_tag = <Tag><size_t>tag.value
        cdef TagMask cpp_tag_mask = <TagMask><size_t>tag_mask.value
        cdef shared_ptr[TagProbeInfo] probe_info_ptr

        with nogil:
            probe_info_ptr = self._worker.get().tagProbe(cpp_tag, cpp_tag_mask, remove)

        cdef TagProbeResult result = TagProbeResult.__new__(TagProbeResult)
        result._probe_info_ptr = probe_info_ptr
        return result

    def set_progress_thread_start_callback(
            self, cb_func, tuple cb_args=None, dict cb_kwargs=None
    ) -> None:
        if cb_args is None:
            cb_args = ()
        if cb_kwargs is None:
            cb_kwargs = {}

        self._progress_thread_start_cb_data = {
            "cb_func": cb_func,
            "cb_args": cb_args,
            "cb_kwargs": cb_kwargs,
        }

        cdef function[void(void*)]* func_generic_callback = (
            new function[void(void*)](_generic_callback)
        )
        with nogil:
            self._worker.get().setProgressThreadStartCallback(
                deref(func_generic_callback), <void*>self._progress_thread_start_cb_data
            )
        del func_generic_callback

    def stop_request_notifier_thread(self) -> None:
        with nogil:
            self._worker.get().stopRequestNotifierThread()

    def wait_request_notifier(
            self,
            uint64_t
            period_ns=0
    ) -> PythonRequestNotifierWaitState:
        cdef RequestNotifierWaitState state
        cdef uint64_t p = period_ns

        with nogil:
            state = self._worker.get().waitRequestNotifier(p)

        return PythonRequestNotifierWaitState(state)

    def run_request_notifier(self) -> None:
        with nogil:
            self._worker.get().runRequestNotifier()

    def populate_python_futures_pool(self) -> None:
        with nogil:
            self._worker.get().populateFuturesPool()

    def clear_python_futures_pool(self) -> None:
        with nogil:
            self._worker.get().clearFuturesPool()

    def tag_recv(
        self,
        Array arr,
        UCXXTag tag,
        UCXXTagMask tag_mask = UCXXTagMaskFull
    ) -> UCXRequest:
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[Request] req
        cdef unique_ptr[RequestTagBuilder] builder
        cdef Tag cpp_tag = <Tag><size_t>tag.value
        cdef TagMask cpp_tag_mask = <TagMask><size_t>tag_mask.value

        if not self._context_feature_flags & Feature.TAG.value:
            raise ValueError("UCXContext must be created with `Feature.TAG`")

        with nogil:
            builder = make_unique[RequestTagBuilder](
                self._worker.get().tagRecvBuilder(
                    buf, nbytes, cpp_tag, cpp_tag_mask
                )
            )
            builder.get().pythonFuture(self._enable_python_future)
            req = builder.get().build()

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def tag_recv_with_handle(
        self,
        Array arr,
        TagProbeResult probe_result
    ) -> UCXRequest:
        """Receive tag message using message handle obtained from tag_probe_with_handle.

        This is more efficient than regular tag_recv as it doesn't need to go through
        the message matching queue again.
        """
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[Request] req
        cdef unique_ptr[RequestTagBuilder] builder

        if not self._context_feature_flags & Feature.TAG.value:
            raise ValueError("UCXContext must be created with `Feature.TAG`")

        if not probe_result.matched:
            raise ValueError("TagProbeResult must be matched")

        with nogil:
            builder = make_unique[RequestTagBuilder](
                self._worker.get().tagRecvWithHandleBuilder(
                    buf, nbytes, probe_result._probe_info_ptr
                )
            )
            builder.get().pythonFuture(self._enable_python_future)
            req = builder.get().build()

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)


cdef class UCXRequest():
    def __init__(self, uintptr_t shared_ptr_request, bint enable_python_future) -> None:
        self._request = deref(<shared_ptr[Request] *> shared_ptr_request)
        self._enable_python_future = enable_python_future
        self._completed = False

    def __dealloc__(self) -> None:
        with nogil:
            self._request.get().cancel()
            self._request.reset()

    @property
    def ucxx_ptr(self) -> int:
        cdef Request* request

        with nogil:
            request = self._request.get()

        return int(<uintptr_t>request)

    cdef shared_ptr[Request] get_ucxx_shared_ptr(self) nogil:
        return self._request

    @property
    def completed(self) -> bool:
        cdef bint completed

        if self._completed is True:
            return True

        with nogil:
            completed = self._request.get().isCompleted()

        return completed

    @property
    def status(self) -> ucs_status_t:
        cdef ucs_status_t status

        with nogil:
            status = self._request.get().getStatus()

        return status

    @property
    def future(self) -> object:
        cdef PyObject* future_ptr

        with nogil:
            future_ptr = <PyObject*>self._request.get().getFuture()

        return <object>future_ptr

    @property
    def recv_buffer(self) -> object:
        cdef shared_ptr[Buffer] buf
        cdef BufferType bufType

        with nogil:
            buf = self._request.get().getRecvBuffer()
            bufType = buf.get().getType() if buf != nullptr else BufferType.Invalid

        # If buf == NULL, it's not allocated by the request but rather the user
        if buf == NULL:
            return None
        elif bufType == BufferType.CCCL:
            return _get_cccl_buffer(buf)
        elif bufType == BufferType.Host:
            return _get_host_buffer(<uintptr_t><void*>buf.get())

    def check_error(self) -> None:
        with nogil:
            self._request.get().checkError()

    async def wait_yield(self) -> None:
        while True:
            if self.completed:
                return self.check_error()
            await asyncio.sleep(0)

    async def wait(self) -> None:
        if self._enable_python_future:
            await self.future
        else:
            await self.wait_yield()


cdef class UCXBufferRequest:
    def __init__(
            self,
            uintptr_t shared_ptr_buffer_request,
            bint enable_python_future
    ) -> None:
        self._buffer_request = deref(<BufferRequestPtr *> shared_ptr_buffer_request)
        self._enable_python_future = enable_python_future

    def __dealloc__(self) -> None:
        with nogil:
            self._buffer_request.reset()

    @property
    def request(self) -> UCXRequest:
        return UCXRequest(
            <uintptr_t><void*>&self._buffer_request.get().request,
            self._enable_python_future,
        )

    @property
    def py_buffer(self) -> object:
        cdef shared_ptr[Buffer] buf
        cdef BufferType bufType

        with nogil:
            buf = self._buffer_request.get().buffer
            bufType = buf.get().getType() if buf != nullptr else BufferType.Invalid

        # If buf == NULL, it holds a header
        if buf == NULL:
            return None
        elif bufType == BufferType.CCCL:
            return _get_cccl_buffer(buf)
        elif bufType == BufferType.Host:
            return _get_host_buffer(<uintptr_t><void*>buf.get())


cdef class UCXBufferRequests:
    def __init__(
        self,
        uintptr_t unique_ptr_buffer_requests,
        bint enable_python_future
    ) -> None:
        self._enable_python_future = enable_python_future
        self._completed = False
        self._requests = tuple()

        self._ucxx_request_tag_multi = (
            deref(<RequestTagMultiPtr *> unique_ptr_buffer_requests)
        )

    def __dealloc__(self) -> None:
        with nogil:
            self._ucxx_request_tag_multi.reset()

    def _populate_requests(self) -> None:
        cdef vector[BufferRequestPtr] requests
        if len(self._requests) == 0:
            requests = deref(self._ucxx_request_tag_multi)._bufferRequests
            total_requests = requests.size()
            self._buffer_requests = tuple([
                UCXBufferRequest(
                    <uintptr_t><void*>&(requests[i]),
                    self._enable_python_future
                )
                for i in range(total_requests)
            ])

            self._requests = tuple([br.request for br in self._buffer_requests])

    @property
    def completed(self) -> bool:
        cdef bint completed

        if self._completed is False:
            with nogil:
                completed = self._ucxx_request_tag_multi.get().isCompleted()
            self._completed = completed

        return self._completed

    @property
    def all_completed(self) -> bool:
        if self._completed is False:
            if self._ucxx_request_tag_multi.get()._isFilled is False:
                return False

            self._populate_requests()

            self._completed = all(
                [r.completed for r in self._requests]
            )

        return self._completed

    @property
    def status(self) -> ucx_status_t:
        cdef ucs_status_t status

        with nogil:
            status = self._ucxx_request_tag_multi.get().getStatus()

        return status

    @property
    def future(self) -> object:
        cdef PyObject* future_ptr

        with nogil:
            future_ptr = <PyObject*>self._ucxx_request_tag_multi.get().getFuture()

        return <object>future_ptr

    @property
    def py_buffers(self) -> tuple[object, ...]:
        if not self.completed:
            raise RuntimeError("Some requests are not completed yet")

        self._populate_requests()

        py_buffers = [br.py_buffer for br in self._buffer_requests]
        # PyBuffers that are None are headers
        return [b for b in py_buffers if b is not None]

    @property
    def requests(self) -> tuple[UCXRequest, ...]:
        self._populate_requests()
        return self._requests

    def check_error(self) -> None:
        with nogil:
            self._ucxx_request_tag_multi.get().checkError()

    async def wait_yield(self) -> None:
        while True:
            if self.completed:
                for r in self._requests:
                    r.check_error()
                return
            await asyncio.sleep(0)

    async def wait(self) -> None:
        if self._enable_python_future:
            await self.future
        else:
            await self.wait_yield()


cdef void _endpoint_close_callback(ucs_status_t status, shared_ptr[void] args) with gil:
    """Callback function called when UCXEndpoint closes or errors"""
    cdef shared_ptr[uintptr_t] cb_data_ptr = static_pointer_cast[uintptr_t, void](args)
    cdef dict cb_data = <dict><void*>cb_data_ptr.get()[0]

    try:
        cb_data['cb_func'](
            *cb_data['cb_args'],
            **cb_data['cb_kwargs']
        )
    except Exception as e:
        logger.error(f"{type(e)} when calling endpoint close callback: {e}")


cdef class UCXEndpoint():
    def __init__(self) -> None:
        raise TypeError("UCXEndpoint cannot be instantiated directly.")

    def __dealloc__(self) -> None:
        self.remove_close_callback()

        with nogil:
            self._endpoint.reset()

    @classmethod
    def create(
            cls,
            UCXWorker worker,
            str ip_address,
            uint16_t port,
            bint endpoint_error_handling
    ) -> UCXEndpoint:
        cdef UCXEndpoint endpoint = UCXEndpoint.__new__(UCXEndpoint)
        cdef shared_ptr[Context] ucxx_context
        cdef string addr = ip_address.encode("utf-8")
        cdef unique_ptr[EndpointBuilder] builder

        endpoint._enable_python_future = worker.enable_python_future

        with nogil:
            ucxx_context = dynamic_pointer_cast[Context, Component](
                worker._worker.get().getParent()
            )

            endpoint._context_feature_flags = ucxx_context.get().getFeatureFlags()
            endpoint._cuda_support = ucxx_context.get().hasCudaSupport()
            builder = make_unique[EndpointBuilder](
                worker._worker.get().endpointBuilder(addr, port)
            )
            builder.get().endpointErrorHandling(endpoint_error_handling)
            endpoint._endpoint = builder.get().build()

        return endpoint

    @classmethod
    def create_from_conn_request(
            cls,
            UCXListener listener,
            uintptr_t conn_request,
            bint endpoint_error_handling
    ) -> UCXEndpoint:
        cdef UCXEndpoint endpoint = UCXEndpoint.__new__(UCXEndpoint)
        cdef shared_ptr[Context] ucxx_context
        cdef shared_ptr[Worker] ucxx_worker
        cdef unique_ptr[EndpointBuilder] builder

        endpoint._enable_python_future = listener.enable_python_future

        with nogil:
            ucxx_worker = dynamic_pointer_cast[Worker, Component](
                listener._listener.get().getParent()
            )
            ucxx_context = dynamic_pointer_cast[Context, Component](
                ucxx_worker.get().getParent()
            )

            endpoint._context_feature_flags = ucxx_context.get().getFeatureFlags()
            endpoint._cuda_support = ucxx_context.get().hasCudaSupport()
            builder = make_unique[EndpointBuilder](
                listener._listener.get().endpointBuilder(
                    <ucp_conn_request_h>conn_request
                )
            )
            builder.get().endpointErrorHandling(endpoint_error_handling)
            endpoint._endpoint = builder.get().build()

        return endpoint

    @classmethod
    def create_from_worker_address(
            cls,
            UCXWorker worker,
            UCXAddress address,
            bint endpoint_error_handling
    ) -> UCXEndpoint:
        cdef UCXEndpoint endpoint = UCXEndpoint.__new__(UCXEndpoint)
        cdef shared_ptr[Context] ucxx_context
        cdef shared_ptr[Address] ucxx_address = address._address
        cdef unique_ptr[EndpointBuilder] builder

        endpoint._enable_python_future = worker.enable_python_future

        with nogil:
            ucxx_context = dynamic_pointer_cast[Context, Component](
                worker._worker.get().getParent()
            )

            endpoint._context_feature_flags = ucxx_context.get().getFeatureFlags()
            endpoint._cuda_support = ucxx_context.get().hasCudaSupport()
            builder = make_unique[EndpointBuilder](
                worker._worker.get().endpointBuilder(ucxx_address)
            )
            builder.get().endpointErrorHandling(endpoint_error_handling)
            endpoint._endpoint = builder.get().build()

        return endpoint

    @property
    def handle(self) -> int:
        cdef ucp_ep_h handle

        with nogil:
            handle = self._endpoint.get().getHandle()

        return int(<uintptr_t>handle)

    @property
    def ucxx_ptr(self) -> int:
        cdef Endpoint* endpoint

        with nogil:
            endpoint = self._endpoint.get()

        return int(<uintptr_t>endpoint)

    cdef shared_ptr[Endpoint] get_ucxx_shared_ptr(self) nogil:
        return self._endpoint

    @property
    def worker_handle(self) -> int:
        cdef ucp_worker_h handle

        with nogil:
            handle = self._endpoint.get().getWorker().get().getHandle()

        return int(<uintptr_t>handle)

    @property
    def ucxx_worker_ptr(self) -> int:
        cdef Worker* worker

        with nogil:
            worker = self._endpoint.get().getWorker().get()

        return int(<uintptr_t>worker)

    @property
    def alive(self) -> bool:
        cdef bint alive

        with nogil:
            alive = self._endpoint.get().isAlive()

        return alive

    def close(self) -> None:
        cdef shared_ptr[Request] req
        cdef unique_ptr[RequestEndpointCloseBuilder] builder

        with nogil:
            builder = make_unique[RequestEndpointCloseBuilder](
                self._endpoint.get().closeBuilder()
            )
            builder.get().pythonFuture(self._enable_python_future)
            req = builder.get().build()

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def close_blocking(self, uint64_t period=0, uint64_t max_attempts=1) -> None:
        cdef uint64_t c_period = period
        cdef uint64_t c_max_attempts = max_attempts

        with nogil:
            self._endpoint.get().closeBlocking(c_period, c_max_attempts)

    def am_probe(self) -> bool:
        cdef ucp_ep_h handle
        cdef shared_ptr[Worker] worker
        cdef bint ep_matched

        with nogil:
            handle = self._endpoint.get().getHandle()
            worker = self._endpoint.get().getWorker()
            ep_matched = worker.get().amProbe(handle)

        return ep_matched

    def am_send(self, Array arr) -> UCXRequest:
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef bint cuda_array = arr.cuda
        cdef shared_ptr[Request] req
        cdef unique_ptr[RequestAmBuilder] builder

        if not self._context_feature_flags & Feature.AM.value:
            raise ValueError("UCXContext must be created with `Feature.AM`")

        with nogil:
            builder = make_unique[RequestAmBuilder](
                self._endpoint.get().amSendBuilder(
                    buf,
                    nbytes,
                    UCS_MEMORY_TYPE_CUDA if cuda_array else UCS_MEMORY_TYPE_HOST
                )
            )
            builder.get().pythonFuture(self._enable_python_future)
            req = builder.get().build()

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def am_recv(self) -> UCXRequest:
        cdef shared_ptr[Request] req
        cdef unique_ptr[RequestAmBuilder] builder

        if not self._context_feature_flags & Feature.AM.value:
            raise ValueError("UCXContext must be created with `Feature.AM`")

        with nogil:
            builder = make_unique[RequestAmBuilder](
                self._endpoint.get().amRecvBuilder()
            )
            builder.get().pythonFuture(self._enable_python_future)
            req = builder.get().build()

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def stream_send(self, Array arr) -> UCXRequest:
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[Request] req
        cdef unique_ptr[RequestStreamBuilder] builder

        if not self._context_feature_flags & Feature.STREAM.value:
            raise ValueError("UCXContext must be created with `Feature.STREAM`")
        if arr.cuda and not self._cuda_support:
            raise ValueError(
                "UCX is not configured with CUDA support, please ensure that the "
                "available UCX on your environment is built against CUDA and that "
                "`cuda` or `cuda_copy` are present in `UCX_TLS` or that it is using "
                "the default `UCX_TLS=all`."
            )

        with nogil:
            builder = make_unique[RequestStreamBuilder](
                self._endpoint.get().streamSendBuilder(buf, nbytes)
            )
            builder.get().pythonFuture(self._enable_python_future)
            req = builder.get().build()

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def stream_recv(self, Array arr) -> UCXRequest:
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[Request] req
        cdef unique_ptr[RequestStreamBuilder] builder

        if not self._context_feature_flags & Feature.STREAM.value:
            raise ValueError("UCXContext must be created with `Feature.STREAM`")
        if arr.cuda and not self._cuda_support:
            raise ValueError(
                "UCX is not configured with CUDA support, please ensure that the "
                "available UCX on your environment is built against CUDA and that "
                "`cuda` or `cuda_copy` are present in `UCX_TLS` or that it is using "
                "the default `UCX_TLS=all`."
            )

        with nogil:
            builder = make_unique[RequestStreamBuilder](
                self._endpoint.get().streamRecvBuilder(buf, nbytes)
            )
            builder.get().pythonFuture(self._enable_python_future)
            req = builder.get().build()

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def tag_send(self, Array arr, UCXXTag tag) -> UCXRequest:
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[Request] req
        cdef unique_ptr[RequestTagBuilder] builder
        cdef Tag cpp_tag = <Tag><size_t>tag.value

        if not self._context_feature_flags & Feature.TAG.value:
            raise ValueError("UCXContext must be created with `Feature.TAG`")
        if arr.cuda and not self._cuda_support:
            raise ValueError(
                "UCX is not configured with CUDA support, please ensure that the "
                "available UCX on your environment is built against CUDA and that "
                "`cuda` or `cuda_copy` are present in `UCX_TLS` or that it is using "
                "the default `UCX_TLS=all`."
            )

        with nogil:
            builder = make_unique[RequestTagBuilder](
                self._endpoint.get().tagSendBuilder(buf, nbytes, cpp_tag)
            )
            builder.get().pythonFuture(self._enable_python_future)
            req = builder.get().build()

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def tag_recv(
        self,
        Array arr,
        UCXXTag tag,
        UCXXTagMask tag_mask=UCXXTagMaskFull
    ) -> UCXRequest:
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[Request] req
        cdef unique_ptr[RequestTagBuilder] builder
        cdef Tag cpp_tag = <Tag><size_t>tag.value
        cdef TagMask cpp_tag_mask = <TagMask><size_t>tag_mask.value

        if not self._context_feature_flags & Feature.TAG.value:
            raise ValueError("UCXContext must be created with `Feature.TAG`")
        if arr.cuda and not self._cuda_support:
            raise ValueError(
                "UCX is not configured with CUDA support, please ensure that the "
                "available UCX on your environment is built against CUDA and that "
                "`cuda` or `cuda_copy` are present in `UCX_TLS` or that it is using "
                "the default `UCX_TLS=all`."
            )

        with nogil:
            builder = make_unique[RequestTagBuilder](
                self._endpoint.get().tagRecvBuilder(
                    buf, nbytes, cpp_tag, cpp_tag_mask
                )
            )
            builder.get().pythonFuture(self._enable_python_future)
            req = builder.get().build()

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def tag_recv_with_handle(
        self,
        Array arr,
        TagProbeResult probe_result
    ) -> UCXRequest:
        """Receive tag message using message handle obtained from tag_probe_with_handle.

        This is more efficient than regular tag_recv as it doesn't need to go through
        the message matching queue again.
        """
        cdef void* buf = <void*>arr.ptr
        cdef size_t nbytes = arr.nbytes
        cdef shared_ptr[Request] req
        cdef shared_ptr[Worker] worker
        cdef unique_ptr[RequestTagBuilder] builder

        if not self._context_feature_flags & Feature.TAG.value:
            raise ValueError("UCXContext must be created with `Feature.TAG`")

        if not probe_result.matched:
            raise ValueError("TagProbeResult must be matched")

        with nogil:
            worker = self._endpoint.get().getWorker()
            builder = make_unique[RequestTagBuilder](
                worker.get().tagRecvWithHandleBuilder(
                    buf, nbytes, probe_result._probe_info_ptr
                )
            )
            builder.get().pythonFuture(self._enable_python_future)
            req = builder.get().build()

        return UCXRequest(<uintptr_t><void*>&req, self._enable_python_future)

    def tag_send_multi(self, tuple arrays, UCXXTag tag) -> UCXBufferRequests:
        cdef vector[const void*] v_buffer
        cdef vector[size_t] v_size
        cdef vector[int] v_is_cuda
        cdef shared_ptr[Request] ucxx_buffer_requests
        cdef unique_ptr[RequestTagMultiBuilder] builder
        cdef Tag cpp_tag = <Tag><size_t>tag.value

        for arr in arrays:
            if not isinstance(arr, Array):
                raise ValueError(
                    "All elements of the `arrays` should be of `Array` type"
                )
            if arr.cuda and not self._cuda_support:
                raise ValueError(
                    "UCX is not configured with CUDA support, please ensure that the "
                    "available UCX on your environment is built against CUDA and that "
                    "`cuda` or `cuda_copy` are present in `UCX_TLS` or that it is "
                    "using the default `UCX_TLS=all`."
                )

        for arr in arrays:
            v_buffer.push_back(<const void*><uintptr_t>arr.ptr)
            v_size.push_back(arr.nbytes)
            v_is_cuda.push_back(arr.cuda)

        with nogil:
            builder = make_unique[RequestTagMultiBuilder](
                self._endpoint.get().tagMultiSendBuilder(
                    v_buffer, v_size, v_is_cuda, cpp_tag
                )
            )
            builder.get().pythonFuture(self._enable_python_future)
            ucxx_buffer_requests = builder.get().build()

        return UCXBufferRequests(
            <uintptr_t><void*>&ucxx_buffer_requests, self._enable_python_future,
        )

    def tag_recv_multi(
        self,
        UCXXTag tag,
        UCXXTagMask tag_mask = UCXXTagMaskFull,
    ) -> UCXBufferRequests:
        cdef shared_ptr[Request] ucxx_buffer_requests
        cdef unique_ptr[RequestTagMultiBuilder] builder
        cdef Tag cpp_tag = <Tag><size_t>tag.value
        cdef TagMask cpp_tag_mask = <TagMask><size_t>tag_mask.value

        with nogil:
            builder = make_unique[RequestTagMultiBuilder](
                self._endpoint.get().tagMultiRecvBuilder(cpp_tag, cpp_tag_mask)
            )
            builder.get().pythonFuture(self._enable_python_future)
            ucxx_buffer_requests = builder.get().build()

        return UCXBufferRequests(
            <uintptr_t><void*>&ucxx_buffer_requests, self._enable_python_future,
        )

    def raise_on_error(self) -> None:
        with nogil:
            self._endpoint.get().raiseOnError()

    def set_close_callback(
            self,
            cb_func,
            tuple cb_args=None,
            dict cb_kwargs=None
    ) -> None:
        if cb_args is None:
            cb_args = ()
        if cb_kwargs is None:
            cb_kwargs = {}

        self._close_cb_data = {
            "cb_func": cb_func,
            "cb_args": cb_args,
            "cb_kwargs": cb_kwargs,
        }
        self._close_cb_data_ptr = make_shared[uintptr_t](
            <uintptr_t><void*>self._close_cb_data
        )

        cdef function[void(ucs_status_t, shared_ptr[void])]* func_close_callback = (
            new function[void(ucs_status_t, shared_ptr[void])](_endpoint_close_callback)
        )
        with nogil:
            self._endpoint.get().setCloseCallback(
                deref(func_close_callback),
                static_pointer_cast[void, uintptr_t](self._close_cb_data_ptr)
            )
        del func_close_callback

    def remove_close_callback(self) -> None:
        cdef Endpoint* endpoint

        with nogil:
            # Unset close callback, in case the Endpoint error callback runs
            # after the Python object has been destroyed.
            # Cast explicitly to prevent Cython `Cannot assign type ...` errors.
            endpoint = self._endpoint.get()
            if endpoint != nullptr:
                endpoint.setCloseCallback(
                    <function[void (ucs_status_t, shared_ptr[void]) except *]>nullptr,
                    <shared_ptr[void]>nullptr,
                )


cdef void _listener_callback(ucp_conn_request_h conn_request, void *args) with gil:
    """Callback function used by UCXListener"""
    cdef dict cb_data = <dict> args

    try:
        cb_data['cb_func'](
            (
                cb_data['listener']().create_endpoint_from_conn_request(
                    int(<uintptr_t>conn_request), True
                ) if 'listener' in cb_data else
                int(<uintptr_t>conn_request)
            ),
            *cb_data['cb_args'],
            **cb_data['cb_kwargs']
        )
    except Exception as e:
        logger.error(f"{type(e)} when calling listener callback: {e}")


cdef class UCXListener():
    def __init__(self) -> None:
        raise TypeError("UCXListener cannot be instantiated directly.")

    def __dealloc__(self) -> None:
        with nogil:
            self._listener.reset()

    @classmethod
    def create(
            cls,
            UCXWorker worker,
            uint16_t port,
            cb_func,
            tuple cb_args=None,
            dict cb_kwargs=None,
            bint deliver_endpoint=False,
    ) -> UCXListener:
        if cb_args is None:
            cb_args = ()
        if cb_kwargs is None:
            cb_kwargs = {}

        cdef UCXListener listener = UCXListener.__new__(UCXListener)
        cdef ucp_listener_conn_callback_t listener_cb = (
            <ucp_listener_conn_callback_t>_listener_callback
        )
        cdef unique_ptr[ListenerBuilder] builder
        cdef dict cb_data = {
            "cb_func": cb_func,
            "cb_args": cb_args,
            "cb_kwargs": cb_kwargs,
        }
        if deliver_endpoint is True:
            cb_data["listener"] = weakref.ref(listener)

        listener._cb_data = cb_data
        listener._enable_python_future = worker.enable_python_future

        with nogil:
            builder = make_unique[ListenerBuilder](
                worker._worker, port, listener_cb, <void*>listener._cb_data
            )
            listener._listener = builder.get().build()

        return listener

    @property
    def ucxx_ptr(self) -> int:
        cdef Listener* listener

        with nogil:
            listener = self._listener.get()

        return int(<uintptr_t>listener)

    cdef shared_ptr[Listener] get_ucxx_shared_ptr(self) nogil:
        return self._listener

    @property
    def port(self) -> int:
        cdef uint16_t port

        with nogil:
            port = self._listener.get().getPort()

        return port

    @property
    def ip(self) -> str:
        cdef string ip

        with nogil:
            ip = self._listener.get().getIp()

        return ip.decode("utf-8")

    @property
    def enable_python_future(self) -> bool:
        return self._enable_python_future

    def create_endpoint_from_conn_request(
            self,
            uintptr_t conn_request,
            bint endpoint_error_handling
    ) -> UCXEndpoint:
        return UCXEndpoint.create_from_conn_request(
            self, conn_request, endpoint_error_handling,
        )


def get_current_options() -> dict:
    """
    Returns the current UCX options
    if UCX were to be initialized now.
    """
    return UCXConfig().config


def get_ucx_version() -> tuple[int, int, int]:
    cdef unsigned int a, b, c
    ucp_get_version(&a, &b, &c)
    return (a, b, c)
