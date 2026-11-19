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
#include <ucxx/request_mem.h>

namespace ucxx {

std::shared_ptr<RequestMem> detail::ConstructorFactory::createRequestMem(
  std::shared_ptr<Endpoint> endpoint,
  const std::variant<data::MemPut, data::MemGet> requestData,
  const bool enablePythonFuture,
  RequestCallbackUserFunction callbackFunction,
  RequestCallbackUserData callbackData)
{
  std::shared_ptr<RequestMem> req = std::visit(
    data::dispatch{
      [&endpoint, &enablePythonFuture, &callbackFunction, &callbackData](data::MemPut memPut) {
        return std::shared_ptr<RequestMem>(new RequestMem(endpoint,
                                                          memPut,
                                                          std::move("memPut"),
                                                          enablePythonFuture,
                                                          callbackFunction,
                                                          callbackData));
      },
      [&endpoint, &enablePythonFuture, &callbackFunction, &callbackData](data::MemGet memGet) {
        return std::shared_ptr<RequestMem>(new RequestMem(endpoint,
                                                          memGet,
                                                          std::move("memGet"),
                                                          enablePythonFuture,
                                                          callbackFunction,
                                                          callbackData));
      },
    },
    requestData);

  // A delayed notification request is not populated immediately, instead it is
  // delayed to allow the worker progress thread to set its status, and more
  // importantly the Python future later on, so that we don't need the GIL here.
  req->_worker->registerDelayedSubmission(
    req, std::bind(std::mem_fn(&Request::populateDelayedSubmission), req.get()));

  return req;
}

RequestMem::RequestMem(std::shared_ptr<Endpoint> endpoint,
                       const std::variant<data::MemPut, data::MemGet> requestData,
                       std::string operationName,
                       const bool enablePythonFuture,
                       RequestCallbackUserFunction callbackFunction,
                       RequestCallbackUserData callbackData)
  : Request(endpoint,
            data::getRequestData(requestData),
            std::move(operationName),
            enablePythonFuture,
            callbackFunction,
            callbackData)
{
  std::visit(data::dispatch{
               [this](data::MemPut) {
                 if (_endpoint == nullptr)
                   throw ucxx::Error("A valid endpoint is required to send memory messages.");
               },
               [this](data::MemGet) {
                 if (_endpoint == nullptr)
                   throw ucxx::Error("A valid endpoint is required to receive memory messages.");
               },
               [](auto) { throw std::runtime_error("Unreachable"); },
             },
             requestData);
}

void RequestMem::memPutCallback(void* request, ucs_status_t status, void* arg)
{
  Request* req = reinterpret_cast<Request*>(arg);
  ucxx_trace_req_f(req->getOwnerString().c_str(), nullptr, request, "memPut", "memPutCallback");
  return req->callback(request, status);
}

void RequestMem::memGetCallback(void* request, ucs_status_t status, void* arg)
{
  Request* req = reinterpret_cast<Request*>(arg);
  ucxx_trace_req_f(req->getOwnerString().c_str(), nullptr, request, "memGet", "memGetCallback");
  return req->callback(request, status);
}

void RequestMem::request()
{
  ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                               UCP_OP_ATTR_FIELD_FLAGS |
                                               UCP_OP_ATTR_FIELD_USER_DATA,
                               .flags     = UCP_AM_SEND_FLAG_REPLY,
                               .datatype  = ucp_dt_make_contig(1),
                               .user_data = this};

  void* request = nullptr;

  std::visit(data::dispatch{
               [this, &request, &param](data::MemPut memPut) {
                 param.cb.send = memPutCallback;
                 request       = ucp_put_nbx(_endpoint->getHandle(),
                                       memPut._buffer,
                                       memPut._length,
                                       memPut._remoteAddr,
                                       memPut._rkey,
                                       &param);
               },
               [this, &request, &param](data::MemGet memGet) {
                 param.cb.send = memGetCallback;
                 request       = ucp_get_nbx(_endpoint->getHandle(),
                                       memGet._buffer,
                                       memGet._length,
                                       memGet._remoteAddr,
                                       memGet._rkey,
                                       &param);
               },
               [](auto) { throw std::runtime_error("Unreachable"); },
             },
             _requestData);

  publishRequest(request);
}

void RequestMem::populateDelayedSubmissionImpl()
{
  bool terminate =
    std::visit(data::dispatch{
                 [this](data::MemPut) {
                   if (_endpoint->getHandle() == nullptr) {
                     ucxx_warn("Endpoint was closed before message could be sent");
                     Request::callback(this, UCS_ERR_CANCELED);
                     return true;
                   }
                   return false;
                 },
                 [this](data::MemGet) {
                   if (_worker->getHandle() == nullptr) {
                     ucxx_warn("Endpoint was closed before message could be received");
                     Request::callback(this, UCS_ERR_CANCELED);
                     return true;
                   }
                   return false;
                 },
                 [](auto) -> decltype(terminate) { throw std::runtime_error("Unreachable"); },
               },
               _requestData);
  if (terminate) return;

  request();

  auto log =
    [this](
      const void* buffer, const size_t length, const uint64_t remoteAddr, const ucp_rkey_h rkey) {
      if (_enablePythonFuture)
        ucxx_trace_req_f(
          _ownerString.c_str(),
          this,
          _request,
          _operationName.c_str(),
          "populateDelayedSubmission, buffer: %p, size: %lu, remoteAddr: 0x%lx, rkey: %p, "
          "future: %p, future handle: %p",
          buffer,
          length,
          remoteAddr,
          rkey,
          _future.get(),
          _future->getHandle());
      else
        ucxx_trace_req_f(
          _ownerString.c_str(),
          this,
          _request,
          _operationName.c_str(),
          "populateDelayedSubmission, buffer: %p, size: %lu, remoteAddr: 0x%lx, rkey: %p",
          buffer,
          length,
          remoteAddr,
          rkey);
    };

  std::visit(data::dispatch{
               [this, &log](data::MemPut memPut) {
                 log(memPut._buffer, memPut._length, memPut._remoteAddr, memPut._rkey);
               },
               [this, &log](data::MemGet memGet) {
                 log(memGet._buffer, memGet._length, memGet._remoteAddr, memGet._rkey);
               },
               [](auto) { throw std::runtime_error("Unreachable"); },
             },
             _requestData);

  process();
}

}  // namespace ucxx
