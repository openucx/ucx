# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

# distutils: language = c++
# cython: language_level=3

from cpython.ref cimport PyObject
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.utility cimport move

from .python_future_task_api cimport *


cdef class PythonFutureTaskApplication():
    cdef unique_ptr[Application] _application

    def __init__(self, asyncio_event_loop):
        cdef PyObject* asyncio_event_loop_ptr = <PyObject*>asyncio_event_loop

        with nogil:
            self._application = move(make_unique[Application](asyncio_event_loop_ptr))

    def submit(self, duration=1.0, id=0):
        cdef double cpp_duration = duration
        cdef long long cpp_id = id
        cdef PyObject* future_ptr

        with nogil:
            future_ptr = self._application.get().submit(cpp_duration, cpp_id)

        return <object>future_ptr
