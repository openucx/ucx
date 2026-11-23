/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <internal/constructors.h>
#include <ucxx/detail/builder_utils.h>
#include <ucxx/endpoint_builder.h>

namespace ucxx {

enum class EndpointBuilderSource { Hostname, ConnRequest, WorkerAddress };

struct EndpointBuilder::Impl {
  EndpointBuilderSource source;
  std::shared_ptr<Worker> worker{nullptr};
  std::shared_ptr<Listener> listener{nullptr};
  std::shared_ptr<Address> address{nullptr};
  std::string ipAddress{};
  uint16_t port{0};
  ucp_conn_request_h connRequest{nullptr};
  bool endpointErrorHandling{true};

  Impl(std::shared_ptr<Worker> w, std::string ip, uint16_t p)
    : source(EndpointBuilderSource::Hostname),
      worker(std::move(w)),
      ipAddress(std::move(ip)),
      port(p)
  {
  }

  Impl(std::shared_ptr<Listener> l, ucp_conn_request_h conn)
    : source(EndpointBuilderSource::ConnRequest), listener(std::move(l)), connRequest(conn)
  {
  }

  Impl(std::shared_ptr<Worker> w, std::shared_ptr<Address> a)
    : source(EndpointBuilderSource::WorkerAddress), worker(std::move(w)), address(std::move(a))
  {
  }
};

EndpointBuilder::EndpointBuilder(std::shared_ptr<Worker> worker,
                                 std::string ipAddress,
                                 uint16_t port)
  : _impl(std::make_unique<Impl>(std::move(worker), std::move(ipAddress), port))
{
}

EndpointBuilder::EndpointBuilder(std::shared_ptr<Listener> listener, ucp_conn_request_h connRequest)
  : _impl(std::make_unique<Impl>(std::move(listener), connRequest))
{
}

EndpointBuilder::EndpointBuilder(std::shared_ptr<Worker> worker, std::shared_ptr<Address> address)
  : _impl(std::make_unique<Impl>(std::move(worker), std::move(address)))
{
}

UCXX_BUILDER_PIMPL_DEFAULTS(EndpointBuilder, Endpoint)

EndpointBuilder& EndpointBuilder::endpointErrorHandling(bool enable)
{
  _impl->endpointErrorHandling = enable;
  return *this;
}

std::shared_ptr<Endpoint> EndpointBuilder::build()
{
  switch (_impl->source) {
    case EndpointBuilderSource::Hostname:
      return detail::ConstructorFactory::createEndpointFromHostname(
        _impl->worker, _impl->ipAddress, _impl->port, _impl->endpointErrorHandling);
    case EndpointBuilderSource::ConnRequest:
      return detail::ConstructorFactory::createEndpointFromConnRequest(
        _impl->listener, _impl->connRequest, _impl->endpointErrorHandling);
    case EndpointBuilderSource::WorkerAddress:
      return detail::ConstructorFactory::createEndpointFromWorkerAddress(
        _impl->worker, _impl->address, _impl->endpointErrorHandling);
  }

  throw std::logic_error("Invalid EndpointBuilder source");
}

}  // namespace ucxx
