/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <ucxx/log.h>

#include <Python.h>

namespace ucxx {

namespace python {

PyObject* asyncio_str              = NULL;
PyObject* asyncio_future_object    = NULL;
PyObject* call_soon_threadsafe_str = NULL;
PyObject* create_future_str        = NULL;
PyObject* future_str               = NULL;
PyObject* set_exception_str        = NULL;
PyObject* set_result_str           = NULL;
PyObject* done_str                 = NULL;
PyObject* cancelled_str            = NULL;

static int intern_strings(void)
{
  asyncio_str = PyUnicode_InternFromString("asyncio");
  if (asyncio_str == NULL) { return -1; }
  call_soon_threadsafe_str = PyUnicode_InternFromString("call_soon_threadsafe");
  if (call_soon_threadsafe_str == NULL) { return -1; }
  create_future_str = PyUnicode_InternFromString("create_future");
  if (create_future_str == NULL) { return -1; }
  future_str = PyUnicode_InternFromString("Future");
  if (future_str == NULL) { return -1; }
  set_exception_str = PyUnicode_InternFromString("set_exception");
  if (set_exception_str == NULL) { return -1; }
  set_result_str = PyUnicode_InternFromString("set_result");
  if (set_result_str == NULL) { return -1; }
  done_str = PyUnicode_InternFromString("done");
  if (done_str == NULL) { return -1; }
  cancelled_str = PyUnicode_InternFromString("cancelled");
  if (cancelled_str == NULL) { return -1; }
  return 0;
}

static int init_ucxx_python()
{
  if (intern_strings() < 0) goto err;

  return 0;

err:
  if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "could not initialize  Python C-API.");
  return -1;
}

static PyObject* get_asyncio_future_object()
{
  PyObject* asyncio_module = NULL;

  if (asyncio_future_object) return asyncio_future_object;

  PyGILState_STATE state = PyGILState_Ensure();

  if (init_ucxx_python() < 0) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "could not allocate internals.");
    goto finish;
  }

  asyncio_module = PyImport_Import(asyncio_str);
  if (PyErr_Occurred()) ucxx_error("ucxx::python::%s, error importing asyncio", __func__);
  if (asyncio_module == NULL) goto finish;

  asyncio_future_object = PyObject_GetAttr(asyncio_module, future_str);
  if (PyErr_Occurred())
    ucxx_error("ucxx::python::%s, error getting asyncio.Future method", __func__);
  Py_DECREF(asyncio_module);
  if (asyncio_future_object == NULL) { goto finish; }

finish:
  PyGILState_Release(state);
  return asyncio_future_object;
}

PyObject* create_python_future()
{
  PyObject* future_object = NULL;
  PyObject* result        = NULL;

  PyGILState_STATE state = PyGILState_Ensure();

  if (init_ucxx_python() < 0) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "could not allocate internals.");
    goto finish;
  }

  future_object = get_asyncio_future_object();
  if (future_object == NULL) { goto finish; }
  if (!PyCallable_Check(future_object)) {
    PyErr_Format(PyExc_RuntimeError, "%U.%U is not callable.", asyncio_str, future_str);
    goto finish;
  }

  result = PyObject_CallFunctionObjArgs(future_object, NULL);
  if (PyErr_Occurred()) ucxx_error("ucxx::python::%s, error creating asyncio.Future", __func__);

finish:
  PyGILState_Release(state);
  return result;
}

PyObject* check_future_state(PyObject* future)
{
  PyObject* result = NULL;

  PyGILState_STATE state = PyGILState_Ensure();

  result =
    PyObject_VectorcallMethod(cancelled_str, &future, 1 | PY_VECTORCALL_ARGUMENTS_OFFSET, NULL);
  if (PyErr_Occurred()) {
    ucxx_error("ucxx::python::%s, error calling `cancelled()` from `asyncio.Future` object",
               __func__);
  } else if (PyObject_IsTrue(result)) {
    ucxx_trace_req("ucxx::python::%s, `asyncio.Future` object has been cancelled.", __func__);
    goto finish;
  }

  result = PyObject_VectorcallMethod(done_str, &future, 1 | PY_VECTORCALL_ARGUMENTS_OFFSET, NULL);
  if (PyErr_Occurred()) {
    ucxx_error("ucxx::python::%s, error calling `done()` from `asyncio.Future` object", __func__);
  } else if (PyObject_IsTrue(result)) {
    ucxx_trace_req("ucxx::python::%s, `asyncio.Future` object is already done.", __func__);
    goto finish;
  }

finish:
  PyGILState_Release(state);

  return result;
}

PyObject* future_set_result(PyObject* future, PyObject* value)
{
  PyObject* result = NULL;

  PyGILState_STATE state = PyGILState_Ensure();

  if (PyObject_IsTrue(check_future_state(future))) {
    ucxx_trace_req(
      "ucxx::python::%s, `asyncio.Future` object is already done or has been cancelled, "
      "skipping `set_result()`.",
      __func__);
    goto finish;
  }

  {
    PyObject* args[] = {future, value};
    result =
      PyObject_VectorcallMethod(set_result_str, args, 2 | PY_VECTORCALL_ARGUMENTS_OFFSET, NULL);
  }
  if (PyErr_Occurred()) {
    ucxx_error("ucxx::python::%s, error calling `set_result()` from `asyncio.Future` object",
               __func__);
    PyErr_Print();
  }

finish:
  PyGILState_Release(state);

  return result;
}

PyObject* future_set_exception(PyObject* future, PyObject* exception, const char* message)
{
  PyObject* result           = NULL;
  PyObject* message_object   = NULL;
  PyObject* message_tuple    = NULL;
  PyObject* formed_exception = NULL;

  PyGILState_STATE state = PyGILState_Ensure();

  if (PyObject_IsTrue(check_future_state(future))) {
    ucxx_trace_req(
      "ucxx::python::%s, `asyncio.Future` object is already done or has been cancelled, "
      "skipping `set_exception()`.",
      __func__);
    goto finish;
  }

  message_object = PyUnicode_FromString(message);
  if (message_object == NULL) goto err;
  message_tuple = PyTuple_Pack(1, message_object);
  if (message_tuple == NULL) goto err;
  formed_exception = PyObject_Call(exception, message_tuple, NULL);
  if (formed_exception == NULL) goto err;

  {
    PyObject* args[] = {future, formed_exception};
    result =
      PyObject_VectorcallMethod(set_exception_str, args, 2 | PY_VECTORCALL_ARGUMENTS_OFFSET, NULL);
  }

  goto finish;

err:
  PyErr_Format(PyExc_RuntimeError, "Error while setting exception for `asyncio.Future`.");
finish:
  Py_XDECREF(message_object);
  Py_XDECREF(message_tuple);
  Py_XDECREF(formed_exception);
  PyGILState_Release(state);
  return result;
}

PyObject* create_python_future_with_event_loop(PyObject* event_loop)
{
  PyObject* result = NULL;

  PyGILState_STATE state = PyGILState_Ensure();

  if (init_ucxx_python() < 0) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "could not allocate internals.");
    goto finish;
  }

  result = PyObject_CallMethodObjArgs(event_loop, create_future_str, NULL);
  if (PyErr_Occurred()) {
    ucxx_error("ucxx::python::%s, error calling `create_future` from event loop object", __func__);
    PyErr_Print();
  }

finish:
  PyGILState_Release(state);
  return result;
}

PyObject* future_set_result_with_event_loop(PyObject* event_loop, PyObject* future, PyObject* value)
{
  PyObject* result              = NULL;
  PyObject* set_result_callable = NULL;

  PyGILState_STATE state = PyGILState_Ensure();

  if (init_ucxx_python() < 0) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "could not allocate internals.");
    goto finish;
  }

  if (event_loop == NULL || !PyObject_TypeCheck(event_loop, &PyBaseObject_Type)) {
    ucxx_error("ucxx::python::%s, invalid or NULL event loop", __func__);
    goto finish;
  }

  set_result_callable = PyObject_GetAttr(future, set_result_str);
  if (PyErr_Occurred()) {
    ucxx_error("ucxx::python::%s, error getting `set_result` method from `asyncio.Future` object",
               __func__);
    PyErr_Print();
    goto finish;
  }
  if (!PyCallable_Check(set_result_callable)) {
    PyErr_Format(PyExc_RuntimeError, "%R.%U is not callable.", future, set_result_str);
    goto finish;
  }

  result = PyObject_CallMethodObjArgs(
    event_loop, call_soon_threadsafe_str, set_result_callable, value, NULL);
  if (PyErr_Occurred()) {
    ucxx_error(
      "ucxx::python::%s, error calling `call_soon_threadsafe` from event loop object to set future "
      "result",
      __func__);
    PyErr_Print();
  }

finish:
  Py_XDECREF(set_result_callable);
  PyGILState_Release(state);
  return result;
}

PyObject* future_set_exception_with_event_loop(PyObject* event_loop,
                                               PyObject* future,
                                               PyObject* exception,
                                               const char* message)
{
  PyObject* result                 = NULL;
  PyObject* set_exception_callable = NULL;
  PyObject* message_object         = NULL;
  PyObject* message_tuple          = NULL;
  PyObject* formed_exception       = NULL;

  PyGILState_STATE state = PyGILState_Ensure();

  if (init_ucxx_python() < 0) {
    if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "could not allocate internals.");
    goto finish;
  }

  if (event_loop == NULL || !PyObject_TypeCheck(event_loop, &PyBaseObject_Type)) {
    ucxx_error("ucxx::python::%s, invalid or NULL event loop", __func__);
    goto finish;
  }

  set_exception_callable = PyObject_GetAttr(future, set_exception_str);
  if (PyErr_Occurred()) {
    ucxx_error(
      "ucxx::python::%s, Error getting `set_exception` method from `asyncio.Future` object",
      __func__);
    PyErr_Print();
    goto finish;
  }
  if (!PyCallable_Check(set_exception_callable)) {
    PyErr_Format(PyExc_RuntimeError, "%R.%U is not callable.", future, set_exception_str);
    goto finish;
  }

  message_object = PyUnicode_FromString(message);
  if (message_object == NULL) goto err;
  message_tuple = PyTuple_Pack(1, message_object);
  if (message_tuple == NULL) goto err;
  formed_exception = PyObject_Call(exception, message_tuple, NULL);
  if (formed_exception == NULL) goto err;

  result = PyObject_CallMethodObjArgs(
    event_loop, call_soon_threadsafe_str, set_exception_callable, formed_exception, NULL);
  if (PyErr_Occurred()) {
    ucxx_error(
      "ucxx::python::%s, Error calling `call_soon_threadsafe` from event loop object to set future "
      "exception",
      __func__);
    PyErr_Print();
  }
  goto finish;

err:
  PyErr_Format(PyExc_RuntimeError,
               "Error while forming exception for `asyncio.Future.set_exception`.");
finish:
  Py_XDECREF(message_object);
  Py_XDECREF(message_tuple);
  Py_XDECREF(formed_exception);
  Py_XDECREF(set_exception_callable);
  PyGILState_Release(state);
  return result;
}

}  // namespace python

}  // namespace ucxx
