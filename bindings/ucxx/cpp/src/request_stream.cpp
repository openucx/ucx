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
#include <ucxx/delayed_submission.h>
#include <ucxx/request_stream.h>

namespace ucxx {
RequestStream::RequestStream(std::shared_ptr<Endpoint> endpoint,
                             const std::variant<data::StreamSend, data::StreamReceive> requestData,
                             std::string operationName,
                             const bool enablePythonFuture)
  : Request(
      endpoint, data::getRequestData(requestData), std::move(operationName), enablePythonFuture)
{
  std::visit(data::dispatch{
               [this](data::StreamSend) {
                 if (_endpoint == nullptr)
                   throw ucxx::Error("A valid endpoint is required to send stream messages.");
               },
               [this](data::StreamReceive) {
                 if (_endpoint == nullptr)
                   throw ucxx::Error("A valid endpoint is required to receive stream messages.");
               },
               [](auto) { throw std::runtime_error("Unreachable"); },
             },
             requestData);
}

std::shared_ptr<RequestStream> detail::ConstructorFactory::createRequestStream(
  std::shared_ptr<Endpoint> endpoint,
  const std::variant<data::StreamSend, data::StreamReceive> requestData,
  const bool enablePythonFuture)
{
  std::shared_ptr<RequestStream> req =
    std::visit(data::dispatch{
                 [&endpoint, &enablePythonFuture](data::StreamSend streamSend) {
                   return std::shared_ptr<RequestStream>(new RequestStream(
                     endpoint, streamSend, std::move("streamSend"), enablePythonFuture));
                 },
                 [&endpoint, &enablePythonFuture](data::StreamReceive streamReceive) {
                   return std::shared_ptr<RequestStream>(new RequestStream(
                     endpoint, streamReceive, std::move("streamReceive"), enablePythonFuture));
                 },
                 [](auto) -> decltype(req) { throw std::runtime_error("Unreachable"); },
               },
               requestData);

  // A delayed notification request is not populated immediately, instead it is
  // delayed to allow the worker progress thread to set its status, and more
  // importantly the Python future later on, so that we don't need the GIL here.
  req->_worker->registerDelayedSubmission(
    req, std::bind(std::mem_fn(&Request::populateDelayedSubmission), req.get()));

  return req;
}

void RequestStream::request()
{
  ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                               UCP_OP_ATTR_FIELD_DATATYPE |
                                               UCP_OP_ATTR_FIELD_USER_DATA,
                               .datatype  = ucp_dt_make_contig(1),
                               .user_data = this};
  void* request             = nullptr;

  std::visit(data::dispatch{
               [this, &request, &param](data::StreamSend streamSend) {
                 param.cb.send = streamSendCallback;
                 request       = ucp_stream_send_nbx(
                   _endpoint->getHandle(), streamSend._buffer, streamSend._length, &param);
               },
               [this, &request, &param](data::StreamReceive streamReceive) {
                 param.op_attr_mask |= UCP_OP_ATTR_FIELD_FLAGS;
                 param.flags          = UCP_STREAM_RECV_FLAG_WAITALL;
                 param.cb.recv_stream = streamRecvCallback;
                 request              = ucp_stream_recv_nbx(_endpoint->getHandle(),
                                               streamReceive._buffer,
                                               streamReceive._length,
                                               &streamReceive._lengthReceived,
                                               &param);
               },
               [](auto) { throw std::runtime_error("Unreachable"); },
             },
             _requestData);

  publishRequest(request);
}

void RequestStream::populateDelayedSubmissionImpl()
{
  bool terminate =
    std::visit(data::dispatch{
                 [this](data::StreamSend) {
                   if (_endpoint->getHandle() == nullptr) {
                     ucxx_warn("Endpoint was closed before message could be sent");
                     Request::callback(this, UCS_ERR_CANCELED);
                     return true;
                   }
                   return false;
                 },
                 [this](data::StreamReceive) {
                   if (_worker->getHandle() == nullptr) {
                     ucxx_warn("Worker was closed before message could be received");
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

  auto log = [this](const void* buffer, const size_t length) {
    if (_enablePythonFuture)
      ucxx_trace_req_f(
        _ownerString.c_str(),
        this,
        _request,
        _operationName.c_str(),
        "populateDelayedSubmission, buffer %p, size %lu, future %p, future handle %p",
        buffer,
        length,
        _future.get(),
        _future->getHandle());
    else
      ucxx_trace_req_f(_ownerString.c_str(),
                       this,
                       _request,
                       _operationName.c_str(),
                       "populateDelayedSubmission, buffer %p, size %lu",
                       buffer,
                       length);
  };

  std::visit(
    data::dispatch{
      [this, &log](data::StreamSend streamSend) { log(streamSend._buffer, streamSend._length); },
      [this, &log](data::StreamReceive streamReceive) {
        log(streamReceive._buffer, streamReceive._length);
      },
      [](auto) { throw std::runtime_error("Unreachable"); },
    },
    _requestData);

  process();
}

void RequestStream::callback(void* request, ucs_status_t status, size_t length)
{
  std::visit(data::dispatch{
               [this, &request, &status, &length](data::StreamReceive streamReceive) {
                 status = length == streamReceive._length ? status : UCS_ERR_MESSAGE_TRUNCATED;

                 if (status == UCS_ERR_MESSAGE_TRUNCATED) {
                   const char* fmt = "length mismatch: %llu (got) != %llu (expected)";
                   int charsLen    = std::snprintf(nullptr, 0, fmt, length, streamReceive._length);
                   if (charsLen > 0) {
                     _status_msg = std::string(static_cast<size_t>(charsLen) + 1,
                                               '\0');  // +1 for null terminator
                     std::snprintf(
                       _status_msg.data(), _status_msg.size(), fmt, length, streamReceive._length);
                   }
                 }

                 Request::callback(request, status);
               },
               [](auto) { throw std::runtime_error("Unreachable"); },
             },
             _requestData);
}

void RequestStream::streamSendCallback(void* request, ucs_status_t status, void* arg)
{
  Request* req = reinterpret_cast<Request*>(arg);
  ucxx_trace_req_f(
    req->getOwnerString().c_str(), nullptr, request, "streamSend", "streamSendCallback");
  return req->callback(request, status);
}

void RequestStream::streamRecvCallback(void* request, ucs_status_t status, size_t length, void* arg)
{
  RequestStream* req = reinterpret_cast<RequestStream*>(arg);
  ucxx_trace_req_f(
    req->getOwnerString().c_str(), nullptr, request, "streamRecv", "streamRecvCallback");
  return req->callback(request, status, length);
}

}  // namespace ucxx
