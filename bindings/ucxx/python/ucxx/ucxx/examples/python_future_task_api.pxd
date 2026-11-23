# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

# distutils: language = c++
# cython: language_level=3


cdef extern from "Python.h" nogil:
    ctypedef struct PyObject


cdef extern from "<ucxx/python/exception.h>" namespace "ucxx::python" nogil:
    cdef void raise_py_error()


cdef extern from "python_future_task.h" namespace "ucxx::python_future_task" nogil:
    cdef cppclass Application:
        Application(PyObject* asyncio_event_loop)
        PyObject* submit(double duration, long long id)
        void* getFuture() except +raise_py_error
