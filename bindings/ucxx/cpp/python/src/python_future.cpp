/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <stdexcept>
#include <utility>

#include <Python.h>

#include <ucp/api/ucp.h>

#include <ucxx/log.h>
#include <ucxx/python/exception.h>
#include <ucxx/python/python_future.h>

namespace ucxx {

namespace python {

Future::Future(PyObject* asyncioEventLoop, std::shared_ptr<::ucxx::Notifier> notifier)
  : ::ucxx::Future(notifier),
    _asyncioEventLoop(asyncioEventLoop),
    _handle{asyncioEventLoop == nullptr ? create_python_future()
                                        : create_python_future_with_event_loop(asyncioEventLoop)}
{
  Py_XINCREF(_asyncioEventLoop);
}

std::shared_ptr<::ucxx::Future> createFuture(std::shared_ptr<::ucxx::Notifier> notifier)
{
  return std::shared_ptr<::ucxx::Future>(new ::ucxx::python::Future(nullptr, notifier));
}

std::shared_ptr<::ucxx::Future> createFutureWithEventLoop(
  PyObject* asyncioEventLoop, std::shared_ptr<::ucxx::Notifier> notifier)
{
  if (asyncioEventLoop == nullptr)
    throw std::runtime_error("The asyncio event loop cannot be a nullptr");
  return std::shared_ptr<::ucxx::Future>(new ::ucxx::python::Future(asyncioEventLoop, notifier));
}

Future::~Future()
{
  // Acquire GIL before decrementing reference counts. Segfaults can occur
  // if `Py_XDECREF` is called without the GIL.
  PyGILState_STATE state = PyGILState_Ensure();
  Py_XDECREF(_handle);
  Py_XDECREF(_asyncioEventLoop);
  PyGILState_Release(state);
}

void Future::set(ucs_status_t status)
{
  if (_handle == nullptr) throw std::runtime_error("Invalid object or already released");

  ucxx_trace_req("ucxx::python::Future::%s, Future: %p, _handle: %p, status: %s",
                 __func__,
                 this,
                 _handle,
                 ucs_status_string(status));
  if (status == UCS_OK) {
    if (_asyncioEventLoop == nullptr)
      std::ignore = future_set_result(_handle, Py_True);
    else
      std::ignore = future_set_result_with_event_loop(_asyncioEventLoop, _handle, Py_True);
  } else {
    if (_asyncioEventLoop == nullptr)
      std::ignore = future_set_exception(
        _handle, get_python_exception_from_ucs_status(status), ucs_status_string(status));
    else
      std::ignore =
        future_set_exception_with_event_loop(_asyncioEventLoop,
                                             _handle,
                                             get_python_exception_from_ucs_status(status),
                                             ucs_status_string(status));
  }
}

void Future::notify(ucs_status_t status)
{
  if (_handle == nullptr) throw std::runtime_error("Invalid object or already released");

  auto s = shared_from_this();

  ucxx_trace_req("ucxx::python::Future::%s, Future: %p, shared.get(): %p, handle: %p, notifier: %p",
                 __func__,
                 this,
                 s.get(),
                 _handle,
                 _notifier.get());
  _notifier->scheduleFutureNotify(shared_from_this(), status);
}

void* Future::getHandle()
{
  if (_handle == nullptr) throw std::runtime_error("Invalid object or already released");

  return _handle;
}

void* Future::release()
{
  if (_handle == nullptr) throw std::runtime_error("Invalid object or already released");

  return std::exchange(_handle, nullptr);
}

}  // namespace python

}  // namespace ucxx
