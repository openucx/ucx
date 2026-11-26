/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>

#include <internal/constructors.h>
#include <ucxx/detail/register_inflight_request.h>
#include <ucxx/endpoint.h>
#include <ucxx/request.h>
#include <ucxx/request_flush.h>
#include <ucxx/request_flush_builder.h>
#include <ucxx/worker.h>

namespace ucxx {

RequestFlushBuilder::RequestFlushBuilder(std::shared_ptr<Component> endpointOrWorker,
                                         data::Flush requestData)
  : _endpointOrWorker(std::move(endpointOrWorker)), _requestData(std::move(requestData))
{
}

std::shared_ptr<RequestFlush> RequestFlushBuilder::build()
{
  markBuilt();
  auto req = detail::ConstructorFactory::createRequestFlush(
    _endpointOrWorker, _requestData, _enablePythonFuture, _callbackFunction, _callbackData);
  detail::registerInflightRequest(_endpointOrWorker, req);
  return req;
}

RequestFlushBuilder::operator std::shared_ptr<RequestFlush>() { return build(); }

RequestFlushBuilder::operator std::shared_ptr<Request>() { return build(); }

}  // namespace ucxx
