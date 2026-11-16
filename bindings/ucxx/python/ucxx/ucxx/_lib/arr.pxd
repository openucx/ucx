# SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

# cython: language_level=3


from libc.stdint cimport uintptr_t


cdef class Array:
    cdef readonly uintptr_t ptr
    cdef readonly bint readonly
    cdef readonly object obj

    cdef readonly Py_ssize_t itemsize

    cdef readonly Py_ssize_t ndim
    cdef Py_ssize_t[::1] shape_mv
    cdef Py_ssize_t[::1] strides_mv

    cdef readonly bint cuda

    cpdef bint _c_contiguous(self)
    cpdef bint _f_contiguous(self)
    cpdef bint _contiguous(self)
    cpdef Py_ssize_t _nbytes(self)


cpdef Array asarray(obj)
