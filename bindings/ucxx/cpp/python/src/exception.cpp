/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdio>
#include <ios>
#include <stdexcept>

#include <ucxx/exception.h>
#include <ucxx/python/exception.h>

namespace ucxx {

namespace python {

PyObject* UCXXError;

PyObject* UCXXNoMessageError;
PyObject* UCXXNoResourceError;
PyObject* UCXXIOError;
PyObject* UCXXNoMemoryError;
PyObject* UCXXInvalidParamError;
PyObject* UCXXUnreachableError;
PyObject* UCXXInvalidAddrError;
PyObject* UCXXNotImplementedError;
PyObject* UCXXMessageTruncatedError;
PyObject* UCXXNoProgressError;
PyObject* UCXXBufferTooSmallError;
PyObject* UCXXNoElemError;
PyObject* UCXXSomeConnectsFailedError;
PyObject* UCXXNoDeviceError;
PyObject* UCXXBusyError;
PyObject* UCXXCanceledError;
PyObject* UCXXShmemSegmentError;
PyObject* UCXXAlreadyExistsError;
PyObject* UCXXOutOfRangeError;
PyObject* UCXXTimedOutError;
PyObject* UCXXExceedsLimitError;
PyObject* UCXXUnsupportedError;
PyObject* UCXXRejectedError;
PyObject* UCXXNotConnectedError;
PyObject* UCXXConnectionResetError;
PyObject* UCXXFirstLinkFailureError;
PyObject* UCXXLastLinkFailureError;
PyObject* UCXXFirstEndpointFailureError;
PyObject* UCXXEndpointTimeoutError;
PyObject* UCXXLastEndpointFailureError;

PyObject* UCXXCloseError;
PyObject* UCXXConfigError;

static PyObject* new_exception(PyObject** exception, const char* name, PyObject* base)
{
  constexpr size_t max_len     = 255;
  char qualified_name[max_len] = {0};

  if (*exception == NULL) {
    snprintf(qualified_name, max_len, "ucxx.%s", name);

    *exception = PyErr_NewException(qualified_name, base, NULL);
  }
  return *exception;
}

void create_exceptions()
{
  new_exception(&UCXXError, "UCXXError", NULL);

  new_exception(&UCXXNoMessageError, "UCXXNoMessageError", UCXXError);
  new_exception(&UCXXNoResourceError, "UCXXNoResourceError", UCXXError);
  new_exception(&UCXXIOError, "UCXXIOError", UCXXError);
  new_exception(&UCXXNoMemoryError, "UCXXNoMemoryError", UCXXError);
  new_exception(&UCXXInvalidParamError, "UCXXInvalidParamError", UCXXError);
  new_exception(&UCXXUnreachableError, "UCXXUnreachableError", UCXXError);
  new_exception(&UCXXInvalidAddrError, "UCXXInvalidAddrError", UCXXError);
  new_exception(&UCXXNotImplementedError, "UCXXNotImplementedError", UCXXError);
  new_exception(&UCXXMessageTruncatedError, "UCXXMessageTruncatedError", UCXXError);
  new_exception(&UCXXNoProgressError, "UCXXNoProgressError", UCXXError);
  new_exception(&UCXXBufferTooSmallError, "UCXXBufferTooSmallError", UCXXError);
  new_exception(&UCXXNoElemError, "UCXXNoElemError", UCXXError);
  new_exception(&UCXXSomeConnectsFailedError, "UCXXSomeConnectsFailedError", UCXXError);
  new_exception(&UCXXNoDeviceError, "UCXXNoDeviceError", UCXXError);
  new_exception(&UCXXBusyError, "UCXXBusyError", UCXXError);
  new_exception(&UCXXCanceledError, "UCXXCanceledError", UCXXError);
  new_exception(&UCXXShmemSegmentError, "UCXXShmemSegmentError", UCXXError);
  new_exception(&UCXXAlreadyExistsError, "UCXXAlreadyExistsError", UCXXError);
  new_exception(&UCXXOutOfRangeError, "UCXXOutOfRangeError", UCXXError);
  new_exception(&UCXXTimedOutError, "UCXXTimedOutError", UCXXError);
  new_exception(&UCXXExceedsLimitError, "UCXXExceedsLimitError", UCXXError);
  new_exception(&UCXXUnsupportedError, "UCXXUnsupportedError", UCXXError);
  new_exception(&UCXXRejectedError, "UCXXRejectedError", UCXXError);
  new_exception(&UCXXNotConnectedError, "UCXXNotConnectedError", UCXXError);
  new_exception(&UCXXConnectionResetError, "UCXXConnectionResetError", UCXXError);
  new_exception(&UCXXFirstLinkFailureError, "UCXXFirstLinkFailureError", UCXXError);
  new_exception(&UCXXLastLinkFailureError, "UCXXLastLinkFailureError", UCXXError);
  new_exception(&UCXXFirstEndpointFailureError, "UCXXFirstEndpointFailureError", UCXXError);
  new_exception(&UCXXEndpointTimeoutError, "UCXXEndpointTimeoutError", UCXXError);
  new_exception(&UCXXLastEndpointFailureError, "UCXXLastEndpointFailureError", UCXXError);

  new_exception(&UCXXConfigError, "UCXXConfigError", UCXXError);
  new_exception(&UCXXCloseError, "UCXXCloseError", UCXXError);
}

void raise_py_error()
{
  try {
    throw;
  } catch (const NoMessageError& e) {
    PyErr_SetString(UCXXNoMessageError, e.what());
  } catch (const NoResourceError& e) {
    PyErr_SetString(UCXXNoResourceError, e.what());
  } catch (const IOError& e) {
    PyErr_SetString(UCXXIOError, e.what());
  } catch (const NoMemoryError& e) {
    PyErr_SetString(UCXXNoMemoryError, e.what());
  } catch (const InvalidParamError& e) {
    PyErr_SetString(UCXXInvalidParamError, e.what());
  } catch (const UnreachableError& e) {
    PyErr_SetString(UCXXUnreachableError, e.what());
  } catch (const InvalidAddrError& e) {
    PyErr_SetString(UCXXInvalidAddrError, e.what());
  } catch (const NotImplementedError& e) {
    PyErr_SetString(UCXXNotImplementedError, e.what());
  } catch (const MessageTruncatedError& e) {
    PyErr_SetString(UCXXMessageTruncatedError, e.what());
  } catch (const NoProgressError& e) {
    PyErr_SetString(UCXXNoProgressError, e.what());
  } catch (const BufferTooSmallError& e) {
    PyErr_SetString(UCXXBufferTooSmallError, e.what());
  } catch (const NoElemError& e) {
    PyErr_SetString(UCXXNoElemError, e.what());
  } catch (const SomeConnectsFailedError& e) {
    PyErr_SetString(UCXXSomeConnectsFailedError, e.what());
  } catch (const NoDeviceError& e) {
    PyErr_SetString(UCXXNoDeviceError, e.what());
  } catch (const BusyError& e) {
    PyErr_SetString(UCXXBusyError, e.what());
  } catch (const CanceledError& e) {
    PyErr_SetString(UCXXCanceledError, e.what());
  } catch (const ShmemSegmentError& e) {
    PyErr_SetString(UCXXShmemSegmentError, e.what());
  } catch (const AlreadyExistsError& e) {
    PyErr_SetString(UCXXAlreadyExistsError, e.what());
  } catch (const OutOfRangeError& e) {
    PyErr_SetString(UCXXOutOfRangeError, e.what());
  } catch (const TimedOutError& e) {
    PyErr_SetString(UCXXTimedOutError, e.what());
  } catch (const ExceedsLimitError& e) {
    PyErr_SetString(UCXXExceedsLimitError, e.what());
  } catch (const UnsupportedError& e) {
    PyErr_SetString(UCXXUnsupportedError, e.what());
  } catch (const RejectedError& e) {
    PyErr_SetString(UCXXRejectedError, e.what());
  } catch (const NotConnectedError& e) {
    PyErr_SetString(UCXXNotConnectedError, e.what());
  } catch (const ConnectionResetError& e) {
    PyErr_SetString(UCXXConnectionResetError, e.what());
  } catch (const FirstLinkFailureError& e) {
    PyErr_SetString(UCXXFirstLinkFailureError, e.what());
  } catch (const LastLinkFailureError& e) {
    PyErr_SetString(UCXXLastLinkFailureError, e.what());
  } catch (const FirstEndpointFailureError& e) {
    PyErr_SetString(UCXXFirstEndpointFailureError, e.what());
  } catch (const EndpointTimeoutError& e) {
    PyErr_SetString(UCXXEndpointTimeoutError, e.what());
  } catch (const LastEndpointFailureError& e) {
    PyErr_SetString(UCXXLastEndpointFailureError, e.what());
  } catch (const Error& e) {
    PyErr_SetString(UCXXError, e.what());
  } catch (const std::bad_alloc& e) {
    PyErr_SetString(PyExc_MemoryError, e.what());
  } catch (const std::bad_cast& e) {
    PyErr_SetString(PyExc_TypeError, e.what());
  } catch (const std::bad_typeid& e) {
    PyErr_SetString(PyExc_TypeError, e.what());
  } catch (const std::domain_error& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
  } catch (const std::invalid_argument& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
  } catch (const std::ios_base::failure& e) {
    PyErr_SetString(PyExc_IOError, e.what());
  } catch (const std::out_of_range& e) {
    PyErr_SetString(PyExc_IndexError, e.what());
  } catch (const std::overflow_error& e) {
    PyErr_SetString(PyExc_OverflowError, e.what());
  } catch (const std::range_error& e) {
    PyErr_SetString(PyExc_ArithmeticError, e.what());
  } catch (const std::underflow_error& e) {
    PyErr_SetString(PyExc_ArithmeticError, e.what());
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "Unknown exception");
  }
}

PyObject* get_python_exception_from_ucs_status(ucs_status_t status)
{
  switch (status) {
    case UCS_ERR_CANCELED: return UCXXCanceledError;
    case UCS_ERR_CONNECTION_RESET: return UCXXConnectionResetError;
    case UCS_ERR_MESSAGE_TRUNCATED: return UCXXMessageTruncatedError;
    default: return UCXXError;
  }
}

}  // namespace python

}  // namespace ucxx
