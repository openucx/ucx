/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "ucxx/request_data.h"
#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include <ucp/api/ucp.h>

#include <internal/constructors.h>
#include <ucxx/buffer.h>
#include <ucxx/delayed_submission.h>
#include <ucxx/request_endpoint_close.h>

namespace ucxx {

std::shared_ptr<RequestEndpointClose> detail::ConstructorFactory::createRequestEndpointClose(
  std::shared_ptr<Endpoint> endpoint,
  const data::EndpointClose requestData,
  const bool enablePythonFuture,
  RequestCallbackUserFunction callbackFunction,
  RequestCallbackUserData callbackData)
{
  std::shared_ptr<RequestEndpointClose> req =
    std::shared_ptr<RequestEndpointClose>(new RequestEndpointClose(endpoint,
                                                                   requestData,
                                                                   std::move("endpointClose"),
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

RequestEndpointClose::RequestEndpointClose(std::shared_ptr<Endpoint> endpoint,
                                           const data::EndpointClose requestData,
                                           std::string operationName,
                                           const bool enablePythonFuture,
                                           RequestCallbackUserFunction callbackFunction,
                                           RequestCallbackUserData callbackData)
  : Request(endpoint,
            requestData,
            std::move(operationName),
            enablePythonFuture,
            callbackFunction,
            callbackData)
{
  if (_endpoint == nullptr && _worker == nullptr)
    throw ucxx::Error("A valid endpoint or worker is required for a close operation.");
}

void RequestEndpointClose::endpointCloseCallback(void* request, ucs_status_t status, void* arg)
{
  Request* req = reinterpret_cast<Request*>(arg);
  ucxx_trace_req_f(
    req->getOwnerString().c_str(), nullptr, request, "endpointClose", "endpointCloseCallback");
  return req->callback(request, status);
}

void RequestEndpointClose::request()
{
  void* request = nullptr;

  ucp_request_param_t param = {
    .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA, .user_data = this};
  if (std::get<data::EndpointClose>(_requestData)._force) param.flags = UCP_EP_CLOSE_FLAG_FORCE;
  param.cb.send = endpointCloseCallback;
  if (_endpoint != nullptr)
    request = ucp_ep_close_nbx(_endpoint->getHandle(), &param);
  else
    throw ucxx::Error("A valid endpoint or worker is required for a close operation.");

  publishRequest(request);
}

void RequestEndpointClose::populateDelayedSubmissionImpl()
{
  if (_endpoint != nullptr && _endpoint->getHandle() == nullptr) {
    ucxx_warn("Endpoint is already closed");
    Request::callback(this, UCS_ERR_CANCELED);
    return;
  }

  request();

  if (_enablePythonFuture)
    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "populateDelayedSubmission, endpoint close, future: %p, future handle: %p",
                     _future.get(),
                     _future->getHandle());
  else
    ucxx_trace_req_f(_ownerString.c_str(),
                     this,
                     _request,
                     _operationName.c_str(),
                     "populateDelayedSubmission, endpoint close");

  process();
}

}  // namespace ucxx
