/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <chrono>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/endpoint.h>
#include <ucxx/typedefs.h>
#include <ucxx/utils/ucx.h>

namespace ucxx {

Request::Request(std::shared_ptr<Component> endpointOrWorker,
                 const data::RequestData requestData,
                 std::string operationName,
                 const bool enablePythonFuture,
                 RequestCallbackUserFunction callbackFunction,
                 RequestCallbackUserData callbackData)
  : _requestData(requestData),
    _operationName(std::move(operationName)),
    _enablePythonFuture(enablePythonFuture),
    _callback(callbackFunction),
    _callbackData(callbackData)
{
  _endpoint = std::dynamic_pointer_cast<Endpoint>(endpointOrWorker);
  _worker =
    _endpoint ? _endpoint->getWorker() : std::dynamic_pointer_cast<Worker>(endpointOrWorker);

  if (_worker == nullptr || _worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");
  if (_endpoint != nullptr && _endpoint->getHandle() == nullptr)
    throw ucxx::Error("Endpoint not initialized");

  _enablePythonFuture &= _worker->isFutureEnabled();
  if (_enablePythonFuture) {
    _future = _worker->getFuture();
    ucxx_trace_req_f(
      _ownerString.c_str(), this, _request, _operationName.c_str(), "future: %p", _future.get());
  }

  std::stringstream ss;

  if (_endpoint) {
    setParent(_endpoint);
    ss << "ucxx::Endpoint: " << _endpoint->getHandle();
    ucxx_trace("ucxx::Request created (%s): %p on ucxx::Endpoint: %p",
               _operationName.c_str(),
               this,
               _endpoint.get());
  } else {
    setParent(_worker);
    ss << "ucxx::Worker: " << _worker->getHandle();
    ucxx_trace("ucxx::Request created (%s): %p on ucxx::Worker: %p",
               _operationName.c_str(),
               this,
               _worker.get());
  }

  _ownerString = ss.str();
}

Request::~Request()
{
  ucxx_trace("ucxx::Request destroyed (%s): %p", _operationName.c_str(), this);
}

void Request::cancel()
{
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  if (_status == UCS_INPROGRESS) {
    if (UCS_PTR_IS_ERR(_request)) {
      ucs_status_t status = UCS_PTR_STATUS(_request);
      ucxx_trace_req_f(_ownerString.c_str(),
                       this,
                       _request,
                       _operationName.c_str(),
                       "unprocessed request during cancelation contains error: %d (%s)",
                       status,
                       ucs_status_string(status));
    } else {
      ucxx_trace_req_f(_ownerString.c_str(), this, _request, _operationName.c_str(), "canceling");
      if (_request != nullptr) {
        ucp_request_cancel(_worker->getHandle(), _request);
      } else {
        setStatus(UCS_ERR_CANCELED);
      }
    }
  } else {
    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "already completed with status: %d (%s)",
                     _status,
                     ucs_status_string(_status));
  }
}

void Request::populateDelayedSubmission()
{
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  if (_status != UCS_INPROGRESS) return;
  populateDelayedSubmissionImpl();
}

ucs_status_t Request::getStatus()
{
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  return _status;
}

void* Request::getFuture()
{
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  return _future ? _future->getHandle() : nullptr;
}

void Request::checkError()
{
  std::lock_guard<std::recursive_mutex> lock(_mutex);

  utils::ucsErrorThrow(_status, _status == UCS_ERR_MESSAGE_TRUNCATED ? _status_msg : std::string());
}

bool Request::isCompleted()
{
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  return _status != UCS_INPROGRESS;
}

void Request::callback(void* request, ucs_status_t status)
{
  /**
   * Prevent reference count to self from going to zero and thus cause self to be destroyed
   * while `callback()` executes.
   */
  decltype(shared_from_this()) selfReference = nullptr;
  try {
    selfReference = shared_from_this();
  } catch (std::bad_weak_ptr& exception) {
    ucxx_debug("ucxx::Request: %p destroyed before callback() was executed", this);
    return;
  }
  if (_status != UCS_INPROGRESS)
    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     request,
                     _operationName.c_str(),
                     "has status already set to %d (%s), callback setting %d (%s)",
                     _status,
                     ucs_status_string(_status),
                     status,
                     ucs_status_string(status));

  ucxx_trace_req_f(_ownerString.c_str(), this, request, _operationName.c_str(), "completed");
  setStatus(status);
  ucxx_trace_req_f(
    _ownerString.c_str(), this, request, _operationName.c_str(), "isCompleted: %d", isCompleted());
}

void Request::process()
{
  std::lock_guard<std::recursive_mutex> lock(_mutex);

  ucs_status_t status = UCS_INPROGRESS;

  if (UCS_PTR_IS_ERR(_request)) {
    // Operation errored immediately
    status = UCS_PTR_STATUS(_request);
  } else if (UCS_PTR_IS_PTR(_request)) {
    // Completion will be handled by callback
    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "completion will be handled by callback");
    return;
  } else {
    // Operation completed immediately
    status = UCS_OK;
  }

  ucxx_trace_req_f(_ownerString.c_str(),
                   this,
                   _request,
                   _operationName.c_str(),
                   "status %d (%s)",
                   status,
                   ucs_status_string(status));

  if (status != UCS_OK) {
    ucxx_debug(
      "error on %s with status %d (%s)", _operationName.c_str(), status, ucs_status_string(status));
  } else {
    ucxx_trace_req_f(
      _ownerString.c_str(), this, _request, _operationName.c_str(), "completed immediately");
  }

  setStatus(status);
}

void Request::setStatus(ucs_status_t status)
{
  {
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    auto requestPtr = std::static_pointer_cast<Request>(shared_from_this());
    if (_endpoint != nullptr) _endpoint->removeInflightRequest(requestPtr);
    _worker->removeInflightRequest(requestPtr);

    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "setStatus called with status %d (%s)",
                     status,
                     ucs_status_string(status));

    if (_status != UCS_INPROGRESS)
      ucxx_error(
        "ucxx::Request: %p, setStatus called with status: %d (%s) but status: %d (%s) was "
        "already set",
        this,
        status,
        ucs_status_string(status),
        _status,
        ucs_status_string(_status));
    _status = status;

    if (_enablePythonFuture) {
      auto future = std::static_pointer_cast<ucxx::Future>(_future);
      future->notify(status);
    }

    if (_callback) {
      ucxx_trace_req_f(
        _ownerString.c_str(), this, _request, _operationName.c_str(), "invoking user callback");
      _callback(status, _callbackData);
    }

    // Free the UCP request inside the lock so it is mutually exclusive with
    // `publishRequest()`/`queryRequestAttributes()` on the submit thread.
    if (UCS_PTR_IS_PTR(_request)) {
      ucp_request_free(_request);
      _request = nullptr;
    }
  }
}

const std::string& Request::getOwnerString() const { return _ownerString; }

void Request::publishRequest(void* request)
{
  if (!_worker->isRequestAttributesEnabled()) {
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    _request = request;
    return;
  }

  std::lock_guard<std::recursive_mutex> lock(_mutex);
  _request = request;

  if (_requestAttr.memoryType != UCS_MEMORY_TYPE_UNKNOWN) return;

  ucp_request_attr_t result;

  auto worker_attr = _worker->queryAttributes();

  std::string debugString(worker_attr.maxDebugString, '\0');

  result.field_mask = UCP_REQUEST_ATTR_FIELD_MEM_TYPE | UCP_REQUEST_ATTR_FIELD_INFO_STRING |
                      UCP_REQUEST_ATTR_FIELD_INFO_STRING_SIZE;

  result.debug_string      = debugString.data();
  result.debug_string_size = debugString.size();

  if (UCS_PTR_IS_PTR(_request)) {
    auto queryStatus = ucp_request_query(_request, &result);
    if (queryStatus == UCS_OK && result.debug_string != nullptr) {
      debugString.resize(std::strlen(debugString.c_str()));
      _requestAttr.debugString = std::move(debugString);
      _requestAttr.memoryType  = result.mem_type;
    }
  }
}

Request::Attributes Request::queryAttributes()
{
  if (!_worker->isRequestAttributesEnabled())
    throw ucxx::UnsupportedError(
      "Request attributes querying is disabled on the owning worker; build the worker "
      "with `ucxx::WorkerBuilder::requestAttributes(true)` to enable it");

  std::lock_guard<std::recursive_mutex> lock(_mutex);

  if (_requestAttr.memoryType != UCS_MEMORY_TYPE_UNKNOWN) return _requestAttr;

  throw ucxx::NoElemError(
    "Request attributes are not available for this request: UCX took an inline-completion "
    "path with no queryable UCP request, or the request has not completed yet");
}

std::shared_ptr<Buffer> Request::getRecvBuffer() { return nullptr; }

std::string Request::getRecvHeader() { return {}; }

}  // namespace ucxx
