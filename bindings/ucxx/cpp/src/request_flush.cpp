/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include <ucp/api/ucp.h>

#include <internal/constructors.h>
#include <ucxx/buffer.h>
#include <ucxx/delayed_submission.h>
#include <ucxx/request_flush.h>

namespace ucxx {

std::shared_ptr<RequestFlush> detail::ConstructorFactory::createRequestFlush(
  std::shared_ptr<Component> endpointOrWorker,
  const data::Flush requestData,
  const bool enablePythonFuture,
  RequestCallbackUserFunction callbackFunction,
  RequestCallbackUserData callbackData)
{
  auto req = std::shared_ptr<RequestFlush>(new RequestFlush(endpointOrWorker,
                                                            requestData,
                                                            std::move("flush"),
                                                            enablePythonFuture,
                                                            callbackFunction,
                                                            callbackData));

  // A delayed notification request is not populated immediately, instead it is
  // delayed to allow the worker progress thread to set its status, and more
  // importantly the Python future later on, so that we don't need the GIL here.
  req->_worker->registerDelayedSubmission(
    req, std::bind(std::mem_fn(&Request::populateDelayedSubmission), req.get()));

  return req;
}

RequestFlush::RequestFlush(std::shared_ptr<Component> endpointOrWorker,
                           const data::Flush requestData,
                           std::string operationName,
                           const bool enablePythonFuture,
                           RequestCallbackUserFunction callbackFunction,
                           RequestCallbackUserData callbackData)
  : Request(endpointOrWorker,
            requestData,
            std::move(operationName),
            enablePythonFuture,
            callbackFunction,
            callbackData)
{
  if (_endpoint == nullptr && _worker == nullptr)
    throw ucxx::Error("A valid endpoint or worker is required for a flush operation.");
}

void RequestFlush::flushCallback(void* request, ucs_status_t status, void* arg)
{
  Request* req = reinterpret_cast<Request*>(arg);
  ucxx_trace_req_f(req->getOwnerString().c_str(), nullptr, request, "flush", "flushCallback");
  return req->callback(request, status);
}

void RequestFlush::request()
{
  ucp_request_param_t param = {
    .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA, .user_data = this};

  void* request = nullptr;

  param.cb.send = flushCallback;
  if (_endpoint != nullptr)
    request = ucp_ep_flush_nbx(_endpoint->getHandle(), &param);
  else if (_worker != nullptr)
    request = ucp_worker_flush_nbx(_worker->getHandle(), &param);
  else
    throw ucxx::Error("A valid endpoint or worker is required for a flush operation.");

  publishRequest(request);
}

void RequestFlush::populateDelayedSubmissionImpl()
{
  if (_endpoint != nullptr && _endpoint->getHandle() == nullptr) {
    ucxx_warn("Endpoint was closed before it could be flushed");
    Request::callback(this, UCS_ERR_CANCELED);
    return;
  } else if (_worker != nullptr && _worker->getHandle() == nullptr) {
    ucxx_warn("Worker was closed before it could be flushed");
    Request::callback(this, UCS_ERR_CANCELED);
    return;
  }

  request();

  std::string flushComponent = "unknown";
  if (_endpoint != nullptr)
    flushComponent = "endpoint";
  else if (_worker != nullptr)
    flushComponent = "worker";

  if (_enablePythonFuture)
    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "populateDelayedSubmission, flush (%s), future: %p, future handle: %p",
                     flushComponent.c_str(),
                     _future.get(),
                     _future->getHandle());
  else
    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "populateDelayedSubmission, flush (%s)",
                     flushComponent.c_str());

  process();
}

}  // namespace ucxx
