# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

from libc.stdint cimport uintptr_t
from libcpp cimport nullptr
from libcpp.memory cimport shared_ptr

from ucxx._lib.libucxx cimport *
import ucxx._lib.libucxx as ucx_api


cdef test_context_ucxx_shared_ptr(UCXContext context):
    cdef shared_ptr[Context] ucxx_shared_ptr = context.get_ucxx_shared_ptr()
    assert ucxx_shared_ptr.get() != nullptr
    assert <uintptr_t>ucxx_shared_ptr.get() == <uintptr_t>context.ucxx_ptr


cdef test_address_ucxx_shared_ptr(UCXAddress address):
    cdef shared_ptr[Address] ucxx_shared_ptr = address.get_ucxx_shared_ptr()
    assert ucxx_shared_ptr.get() != nullptr
    assert <uintptr_t>ucxx_shared_ptr.get() == <uintptr_t>address.ucxx_ptr


cdef test_worker_ucxx_shared_ptr(UCXWorker worker):
    cdef shared_ptr[Worker] ucxx_shared_ptr = worker.get_ucxx_shared_ptr()
    assert ucxx_shared_ptr.get() != nullptr
    assert <uintptr_t>ucxx_shared_ptr.get() == <uintptr_t>worker.ucxx_ptr


cdef test_request_ucxx_shared_ptr(UCXRequest request):
    cdef shared_ptr[Request] ucxx_shared_ptr = request.get_ucxx_shared_ptr()
    assert ucxx_shared_ptr.get() != nullptr
    assert <uintptr_t>ucxx_shared_ptr.get() == <uintptr_t>request.ucxx_ptr


cdef test_endpoint_ucxx_shared_ptr(UCXEndpoint endpoint):
    cdef shared_ptr[Endpoint] ucxx_shared_ptr = endpoint.get_ucxx_shared_ptr()
    assert ucxx_shared_ptr.get() != nullptr
    assert <uintptr_t>ucxx_shared_ptr.get() == <uintptr_t>endpoint.ucxx_ptr


cdef test_listener_ucxx_shared_ptr(UCXListener listener):
    cdef shared_ptr[Listener] ucxx_shared_ptr = listener.get_ucxx_shared_ptr()
    assert ucxx_shared_ptr.get() != nullptr
    assert <uintptr_t>ucxx_shared_ptr.get() == <uintptr_t>listener.ucxx_ptr


def cython_test_context_getter():
    feature_flags = [ucx_api.Feature.WAKEUP, ucx_api.Feature.TAG]
    context = ucx_api.UCXContext(feature_flags=tuple(feature_flags))

    test_context_ucxx_shared_ptr(context)

    worker = ucx_api.UCXWorker(context)
    test_worker_ucxx_shared_ptr(worker)

    address = worker.address
    test_address_ucxx_shared_ptr(address)
