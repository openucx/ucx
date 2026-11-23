/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <Python.h>

namespace ucxx {

namespace python {

/**
 * @brief Create a Python asyncio future.
 *
 * Create Python asyncio future, effectively equal to calling `asyncio.Future()` directly
 * in Python.
 *
 * Note that this call will take the Python GIL and requires that the current thread have
 * an asynchronous event loop set.
 *
 * @returns The Python asyncio future object.
 */
[[nodiscard]] PyObject* create_python_future() noexcept;

/**
 * @brief Set the result of a Python future.
 *
 * Set the result of a Python future.
 *
 * Note that this call will take the Python GIL and requires that the current thread have
 * the same asynchronous event loop set as the thread that owns the future.
 *
 * @param[in] future  Python object containing the `_asyncio.Future` object.
 * @param[in] value   Python object containing an arbitrary value to set the future result
 *                    to.
 *
 * @returns The result of the call to `_asyncio.Future.set_result()`.
 */
[[nodiscard]] PyObject* future_set_result(PyObject* future, PyObject* value) noexcept;

/**
 * @brief Set the exception of a Python future.
 *
 * Set the exception of a Python future.
 *
 * Note that this call will take the Python GIL and requires that the current thread have
 * the same asynchronous event loop set as the thread that owns the future.
 *
 * @param[in] future    Python object containing the `_asyncio.Future` object.
 * @param[in] exception a Python exception derived of the `Exception` class.
 * @param[in] message   human-readable error message for the exception.
 *
 * @returns The result of the call to `_asyncio.Future.set_result()`.
 */
[[nodiscard]] PyObject* future_set_exception(PyObject* future,
                                             PyObject* exception,
                                             const char* message) noexcept;

/**
 * @brief Create a Python asyncio future with associated event loop.
 *
 * Create Python asyncio future associated with the event loop passed via the `event_loop`
 * argument, effectively equal to calling `loop.create_future()` directly in Python.
 *
 * Note that this call will take the Python GIL and requires that the current thread have
 * an asynchronous event loop set.
 *
 * @param[in] event_loop  the Python asyncio event loop to which the future will belong to.
 *
 * @returns The Python asyncio future object.
 */
[[nodiscard]] PyObject* create_python_future_with_event_loop(PyObject* event_loop) noexcept;

/**
 * @brief Set the result of a Python future with associated event loop.
 *
 * Schedule setting the result of a Python future in the given event loop using the
 * threadsafe method `event_loop.call_soon_threadsafe`. The event loop given must be the
 * same specified when creating the future object with `create_python_future`.
 *
 * Note that this may be called from any thread and will take the Python GIL to run.
 *
 * @param[in] event_loop  the Python asyncio event loop to which the future will belong to.
 * @param[in] future      Python object containing the `_asyncio.Future` object.
 * @param[in] value       Python object containing an arbitrary value to set the future
 *                        result to.
 *
 * @returns The result of the call to `_asyncio.Future.set_result()`.
 */
[[nodiscard]] PyObject* future_set_result_with_event_loop(PyObject* event_loop,
                                                          PyObject* future,
                                                          PyObject* value) noexcept;

/**
 * @brief Set the exception of a Python future with associated event loop.
 *
 * Schedule setting an exception of a Python future in the given event loop using the
 * threadsafe method `event_loop.call_soon_threadsafe`. The event loop given must be the
 * same specified when creating the future object with `create_python_future`.
 *
 * Note that this may be called from any thread and will take the Python GIL to run.
 *
 * @param[in] event_loop  the Python asyncio event loop to which the future will belong to.
 * @param[in] future      Python object containing the `_asyncio.Future` object.
 * @param[in] exception   a Python exception derived of the `Exception` class.
 * @param[in] message     human-readable error message for the exception.
 *
 * @returns The result of the call to `_asyncio.Future.set_result()`.
 */
[[nodiscard]] PyObject* future_set_exception_with_event_loop(PyObject* event_loop,
                                                             PyObject* future,
                                                             PyObject* exception,
                                                             const char* message) noexcept;

}  // namespace python

}  // namespace ucxx
