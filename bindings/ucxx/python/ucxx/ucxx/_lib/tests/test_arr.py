# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import array
import functools
import io
import mmap
import operator

import pytest

from ucxx._lib.arr import Array

builtin_buffers = [
    b"",
    b"abcd",
    array.array("i", []),
    array.array("i", [0, 1, 2, 3]),
    array.array("I", [0, 1, 2, 3]),
    array.array("f", []),
    array.array("f", [0, 1, 2, 3]),
    array.array("d", [0, 1, 2, 3]),
    memoryview(array.array("B", [0, 1, 2, 3, 4, 5])).cast("B", (3, 2)),
    memoryview(b"abcd"),
    memoryview(bytearray(b"abcd")),
    io.BytesIO(b"abcd").getbuffer(),
    mmap.mmap(-1, 5),
]


@pytest.mark.parametrize("buffer", builtin_buffers)
def test_Array_ptr_builtins(buffer):
    arr = Array(buffer)
    assert arr.ptr != 0


@pytest.mark.parametrize("buffer", builtin_buffers)
def test_Array_readonly_builtins(buffer):
    arr = Array(buffer)
    mv = memoryview(buffer)
    assert arr.readonly == mv.readonly


@pytest.mark.parametrize("buffer", builtin_buffers)
def test_Array_obj_builtins(buffer):
    arr = Array(buffer)
    obj = arr.obj
    while isinstance(obj, memoryview):
        obj = obj.obj
    mv = memoryview(buffer)
    assert obj is mv.obj


@pytest.mark.parametrize("buffer", builtin_buffers)
def test_Array_itemsize_builtins(buffer):
    arr = Array(buffer)
    mv = memoryview(buffer)
    assert arr.itemsize == mv.itemsize


@pytest.mark.parametrize("buffer", builtin_buffers)
def test_Array_ndim_builtins(buffer):
    arr = Array(buffer)
    mv = memoryview(buffer)
    assert arr.ndim == mv.ndim


@pytest.mark.parametrize("buffer", builtin_buffers)
def test_Array_shape_builtins(buffer):
    arr = Array(buffer)
    mv = memoryview(buffer)
    assert arr.shape == mv.shape


@pytest.mark.parametrize("buffer", builtin_buffers)
def test_Array_strides_builtins(buffer):
    arr = Array(buffer)
    mv = memoryview(buffer)
    assert arr.strides == mv.strides


@pytest.mark.parametrize("buffer", builtin_buffers)
def test_Array_nbytes_builtins(buffer):
    arr = Array(buffer)
    mv = memoryview(buffer)
    assert arr.nbytes == mv.nbytes


@pytest.mark.parametrize("buffer", builtin_buffers)
def test_Array_contiguous_builtins(buffer):
    mv = memoryview(buffer)
    arr = Array(buffer)
    assert arr.c_contiguous == mv.c_contiguous
    assert arr.f_contiguous == mv.f_contiguous
    assert arr.contiguous == mv.contiguous

    mv2 = memoryview(buffer)[::2]
    if mv2:
        arr2 = Array(mv2)
        assert arr2.c_contiguous == mv2.c_contiguous
        assert arr2.f_contiguous == mv2.f_contiguous
        assert arr2.contiguous == mv2.contiguous


array_params = [
    ((2, 3), "i4", (12, 4)),
    ((2, 3), "u4", (12, 4)),
    ((2, 3), "f4", (12, 4)),
    ((2, 3), "f8", (24, 8)),
    ((2, 3), "f8", (8, 16)),
]


def create_array(xp, shape, dtype, strides):
    if xp == "cupy":
        iface_prop = "__cuda_array_interface__"
    elif xp == "numpy":
        iface_prop = "__array_interface__"

    xp = pytest.importorskip(xp)

    nelem = functools.reduce(operator.mul, shape, 1)
    data = xp.arange(nelem, dtype=dtype)
    arr = xp.ndarray(shape, dtype, data.data, strides=strides)
    iface = getattr(arr, iface_prop)

    return xp, arr, iface


@pytest.mark.parametrize("xp", ["cupy", "numpy"])
@pytest.mark.parametrize("shape, dtype, strides", array_params)
def test_Array_ndarray_ptr(xp, shape, dtype, strides):
    xp, arr, iface = create_array(xp, shape, dtype, strides)
    arr2 = Array(arr)

    assert arr2.ptr == iface["data"][0]


@pytest.mark.parametrize("xp", ["cupy", "numpy"])
@pytest.mark.parametrize("shape, dtype, strides", array_params)
def test_Array_ndarray_is_cuda(xp, shape, dtype, strides):
    xp, arr, iface = create_array(xp, shape, dtype, strides)
    arr2 = Array(arr)

    is_cuda = xp.__name__ == "cupy"
    assert arr2.cuda == is_cuda


@pytest.mark.parametrize("xp", ["cupy", "numpy"])
@pytest.mark.parametrize("shape, dtype, strides", array_params)
def test_Array_ndarray_nbytes(xp, shape, dtype, strides):
    xp, arr, iface = create_array(xp, shape, dtype, strides)
    arr2 = Array(arr)

    assert arr2.nbytes == arr.nbytes


@pytest.mark.parametrize("xp", ["cupy", "numpy"])
@pytest.mark.parametrize("shape, dtype, strides", array_params)
def test_Array_ndarray_shape(xp, shape, dtype, strides):
    xp, arr, iface = create_array(xp, shape, dtype, strides)
    arr2 = Array(arr)

    assert arr2.shape == arr.shape


@pytest.mark.parametrize("xp", ["cupy", "numpy"])
@pytest.mark.parametrize("shape, dtype, strides", array_params)
def test_Array_ndarray_strides(xp, shape, dtype, strides):
    xp, arr, iface = create_array(xp, shape, dtype, strides)
    arr2 = Array(arr)

    assert arr2.strides == arr.strides


@pytest.mark.parametrize("xp", ["cupy", "numpy"])
@pytest.mark.parametrize("shape, dtype, strides", array_params)
def test_Array_ndarray_contiguous(xp, shape, dtype, strides):
    xp, arr, iface = create_array(xp, shape, dtype, strides)
    arr2 = Array(arr)

    assert arr2.c_contiguous == arr.flags.c_contiguous
    assert arr2.f_contiguous == arr.flags.f_contiguous
    assert arr2.contiguous == (arr.flags.c_contiguous or arr.flags.f_contiguous)
