/**
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <mutex>

#include <Python.h>

#include <ucxx/log.h>
#include <ucxx/python/python_future_task_collector.h>

namespace ucxx {

namespace python {

PythonFutureTaskCollector& PythonFutureTaskCollector::get()
{
  static PythonFutureTaskCollector collector;
  return collector;
}

void PythonFutureTaskCollector::push(PyObject* handle)
{
  std::lock_guard<std::mutex> lock(_mutex);
  _toCollect.push_back(handle);
}

void PythonFutureTaskCollector::collect()
{
  PyGILState_STATE state = PyGILState_Ensure();

  {
    std::lock_guard<std::mutex> lock(_mutex);
    for (auto& handle : _toCollect)
      Py_XDECREF(handle);
    ucxx_trace("ucxx::python::PythonFutureTaskCollector::%s, collected %lu PythonFutureTasks",
               __func__,
               _toCollect.size());
    _toCollect.clear();
  }

  PyGILState_Release(state);
}

PythonFutureTaskCollector::PythonFutureTaskCollector() {}

PythonFutureTaskCollector::~PythonFutureTaskCollector()
{
  {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_toCollect.size() > 0)
      ucxx_warn("Destroying PythonFutureTaskCollector with %lu uncollected tasks",
                _toCollect.size());
  }
}

}  // namespace python

}  // namespace ucxx
