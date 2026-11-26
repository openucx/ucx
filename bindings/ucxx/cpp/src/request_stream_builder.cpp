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
#include <ucxx/request_stream.h>
#include <ucxx/request_stream_builder.h>

namespace ucxx {

RequestStreamBuilder::RequestStreamBuilder(
  std::shared_ptr<Endpoint> endpoint,
  std::variant<data::StreamSend, data::StreamReceive> requestData)
  : _endpoint(std::move(endpoint)), _requestData(std::move(requestData))
{
}

std::shared_ptr<RequestStream> RequestStreamBuilder::build()
{
  markBuilt();
  auto req =
    detail::ConstructorFactory::createRequestStream(_endpoint, _requestData, _enablePythonFuture);
  detail::registerInflightRequest(_endpoint, req);
  return req;
}

RequestStreamBuilder::operator std::shared_ptr<RequestStream>() { return build(); }

RequestStreamBuilder::operator std::shared_ptr<Request>() { return build(); }

}  // namespace ucxx
