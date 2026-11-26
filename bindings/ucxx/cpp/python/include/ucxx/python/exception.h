/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <Python.h>

#include <ucp/api/ucp.h>

namespace ucxx {

namespace python {

extern PyObject* UCXXError;

extern PyObject* UCXXNoMessageError;
extern PyObject* UCXXNoResourceError;
extern PyObject* UCXXIOError;
extern PyObject* UCXXNoMemoryError;
extern PyObject* UCXXInvalidParamError;
extern PyObject* UCXXUnreachableError;
extern PyObject* UCXXInvalidAddrError;
extern PyObject* UCXXNotImplementedError;
extern PyObject* UCXXMessageTruncatedError;
extern PyObject* UCXXNoProgressError;
extern PyObject* UCXXBufferTooSmallError;
extern PyObject* UCXXNoElemError;
extern PyObject* UCXXSomeConnectsFailedError;
extern PyObject* UCXXNoDeviceError;
extern PyObject* UCXXBusyError;
extern PyObject* UCXXCanceledError;
extern PyObject* UCXXShmemSegmentError;
extern PyObject* UCXXAlreadyExistsError;
extern PyObject* UCXXOutOfRangeError;
extern PyObject* UCXXTimedOutError;
extern PyObject* UCXXExceedsLimitError;
extern PyObject* UCXXUnsupportedError;
extern PyObject* UCXXRejectedError;
extern PyObject* UCXXNotConnectedError;
extern PyObject* UCXXConnectionResetError;
extern PyObject* UCXXFirstLinkFailureError;
extern PyObject* UCXXLastLinkFailureError;
extern PyObject* UCXXFirstEndpointFailureError;
extern PyObject* UCXXEndpointTimeoutError;
extern PyObject* UCXXLastEndpointFailureError;

extern PyObject* UCXXCloseError;
extern PyObject* UCXXConfigError;

/**
 * @brief Create Python exceptions.
 *
 * Create UCXX-specific Python exceptions such that they are visible both from C/C++ and
 * Python.
 */
void create_exceptions();

/**
 * @brief Raise a C++ exception in Python.
 *
 * Raise a C++ exception in Python. When a C++ exception occurs, Python needs to be able
 * to be informed of such event and be able to raise a Python exception from it. This
 * function raises both general C++ exceptions, as well as UCXX-specific exceptions.
 *
 * To use this, C++ methods and functions that are exposed to Python via Cython must have
 * a `except +raise_py_error` as a declaration suffix.
 */
void raise_py_error();

/**
 * @brief Get a Python exception from UCS status.
 *
 * Given a UCS status, get a matching Python exception object.
 *
 * @param[in] status UCS status from which to get exception.
 */
[[nodiscard]] PyObject* get_python_exception_from_ucs_status(ucs_status_t status);

}  // namespace python

}  // namespace ucxx
