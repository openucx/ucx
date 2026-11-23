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
#include <ucxx/request_tag_multi.h>
#include <ucxx/request_tag_multi_builder.h>

namespace ucxx {

RequestTagMultiBuilder::RequestTagMultiBuilder(
  std::shared_ptr<Endpoint> endpoint,
  std::variant<data::TagMultiSend, data::TagMultiReceive> requestData)
  : _endpoint(std::move(endpoint)), _requestData(std::move(requestData))
{
}

std::shared_ptr<RequestTagMulti> RequestTagMultiBuilder::build()
{
  markBuilt();
  auto req =
    detail::ConstructorFactory::createRequestTagMulti(_endpoint, _requestData, _enablePythonFuture);
  detail::registerInflightRequest(_endpoint, req);
  return req;
}

RequestTagMultiBuilder::operator std::shared_ptr<RequestTagMulti>() { return build(); }

RequestTagMultiBuilder::operator std::shared_ptr<Request>() { return build(); }

}  // namespace ucxx
