/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>
#include <variant>

#include <internal/constructors.h>
#include <ucxx/detail/register_inflight_request.h>
#include <ucxx/endpoint.h>
#include <ucxx/request.h>
#include <ucxx/request_mem.h>
#include <ucxx/request_mem_builder.h>

namespace ucxx {

RequestMemBuilder::RequestMemBuilder(std::shared_ptr<Endpoint> endpoint,
                                     std::variant<data::MemPut, data::MemGet> requestData)
  : _endpoint(std::move(endpoint)), _requestData(std::move(requestData))
{
}

RequestMemBuilder& RequestMemBuilder::remoteAddressOffset(uint64_t offset) &
{
  _remoteAddressOffset = offset;
  return *this;
}

RequestMemBuilder&& RequestMemBuilder::remoteAddressOffset(uint64_t offset) &&
{
  remoteAddressOffset(offset);
  return std::move(*this);
}

std::shared_ptr<RequestMem> RequestMemBuilder::build()
{
  markBuilt();
  if (auto* memPut = std::get_if<data::MemPut>(&_requestData)) {
    _requestData.emplace<data::MemPut>(
      memPut->_buffer, memPut->_length, memPut->_remoteAddr + _remoteAddressOffset, memPut->_rkey);
  } else {
    auto* memGet = std::get_if<data::MemGet>(&_requestData);
    _requestData.emplace<data::MemGet>(
      memGet->_buffer, memGet->_length, memGet->_remoteAddr + _remoteAddressOffset, memGet->_rkey);
  }
  auto req = detail::ConstructorFactory::createRequestMem(
    _endpoint, _requestData, _enablePythonFuture, _callbackFunction, _callbackData);
  detail::registerInflightRequest(_endpoint, req);
  return req;
}

RequestMemBuilder::operator std::shared_ptr<RequestMem>() { return build(); }

RequestMemBuilder::operator std::shared_ptr<Request>() { return build(); }

}  // namespace ucxx
