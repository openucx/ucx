/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>

#include <internal/constructors.h>
#include <ucxx/detail/register_inflight_request.h>
#include <ucxx/endpoint.h>
#include <ucxx/request.h>
#include <ucxx/request_am.h>
#include <ucxx/request_am_builder.h>

namespace ucxx {

RequestAmBuilder::RequestAmBuilder(std::shared_ptr<Endpoint> endpoint,
                                   std::variant<data::AmSend, data::AmReceive> requestData)
  : _endpoint(std::move(endpoint)), _requestData(std::move(requestData))
{
}

RequestAmBuilder& RequestAmBuilder::receiverCallbackInfo(
  std::optional<AmReceiverCallbackInfo> info) &
{
  auto* amSend = std::get_if<data::AmSend>(&_requestData);
  if (amSend == nullptr)
    throw std::logic_error("receiverCallbackInfo() is only valid for active message sends");

  auto params                 = AmSendParams{};
  params.flags                = amSend->_flags;
  params.datatype             = amSend->_datatype;
  params.memoryType           = amSend->_memoryType;
  params.memoryTypePolicy     = amSend->_memoryTypePolicy;
  params.receiverCallbackInfo = std::move(info);
  params.userHeader           = amSend->_userHeader;

  if (amSend->_datatype == UCP_DATATYPE_IOV) {
    auto iov = amSend->_iov;
    _requestData.template emplace<data::AmSend>(std::move(iov), params);
  } else {
    _requestData.template emplace<data::AmSend>(amSend->_buffer, amSend->_length, params);
  }

  return *this;
}

RequestAmBuilder&& RequestAmBuilder::receiverCallbackInfo(
  std::optional<AmReceiverCallbackInfo> info) &&
{
  receiverCallbackInfo(std::move(info));
  return std::move(*this);
}

std::shared_ptr<RequestAm> RequestAmBuilder::build()
{
  markBuilt();
  auto req = detail::ConstructorFactory::createRequestAm(
    _endpoint, _requestData, _enablePythonFuture, _callbackFunction, _callbackData);
  detail::registerInflightRequest(_endpoint, req);
  return req;
}

RequestAmBuilder::operator std::shared_ptr<RequestAm>() { return build(); }

RequestAmBuilder::operator std::shared_ptr<Request>() { return build(); }

}  // namespace ucxx
