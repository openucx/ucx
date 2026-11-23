/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <netinet/in.h>
#include <string>
#include <utility>

#include <ucp/api/ucp.h>

#include <internal/constructors.h>
#include <ucxx/exception.h>
#include <ucxx/listener.h>

#include <ucxx/utils/callback_notifier.h>
#include <ucxx/utils/sockaddr.h>
#include <ucxx/utils/ucx.h>

namespace ucxx {

Listener::Listener(std::shared_ptr<Worker> worker,
                   std::string host,
                   uint16_t port,
                   ucp_listener_conn_callback_t callback,
                   void* callbackArgs)
{
  if (worker == nullptr || worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");

  ucp_listener_params_t params = {
    .field_mask   = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER,
    .conn_handler = {.cb = callback, .arg = callbackArgs}};
  auto info               = ucxx::utils::get_addrinfo(host.empty() ? nullptr : host.c_str(), port);
  params.sockaddr.addr    = info->ai_addr;
  params.sockaddr.addrlen = info->ai_addrlen;

  utils::ucsErrorThrow(ucp_listener_create(worker->getHandle(), &params, &_handle));
  ucxx_trace("ucxx::Listener created: %p, UCP handle: %p", this, _handle);

  ucp_listener_attr_t attr = {.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR};
  utils::ucsErrorThrow(ucp_listener_query(_handle, &attr));

  char ipString[INET6_ADDRSTRLEN];
  char portString[INET6_ADDRSTRLEN];
  ucxx::utils::sockaddr_get_ip_port_str(&attr.sockaddr, ipString, portString, INET6_ADDRSTRLEN);

  _ip   = std::string(ipString);
  _port = static_cast<uint16_t>(atoi(portString));

  setParent(worker);
}

Listener::~Listener()
{
  auto worker = std::static_pointer_cast<Worker>(_parent);

  if (worker->isProgressThreadRunning()) {
    std::ignore = worker->registerGenericPre([this]() { ucp_listener_destroy(_handle); },
                                             10000000000 /* 10s */);

    std::ignore = worker->registerGenericPost([]() {}, 10000000000 /* 10s */);
  } else {
    ucp_listener_destroy(_handle);
    worker->progress();
  }

  ucxx_trace("ucxx::Listener destroyed: %p, UCP handle: %p", this, _handle);
}

namespace detail {

std::shared_ptr<Listener> ConstructorFactory::createListener(std::shared_ptr<Worker> worker,
                                                             std::string host,
                                                             uint16_t port,
                                                             ucp_listener_conn_callback_t callback,
                                                             void* callbackArgs)
{
  return std::shared_ptr<Listener>(
    new Listener(worker, std::move(host), port, callback, callbackArgs));
}

}  // namespace detail

EndpointBuilder Listener::endpointBuilder(ucp_conn_request_h connRequest)
{
  return EndpointBuilder(std::static_pointer_cast<Listener>(shared_from_this()), connRequest);
}

ucp_listener_h Listener::getHandle() { return _handle; }

uint16_t Listener::getPort() { return _port; }

std::string Listener::getIp() { return _ip; }

}  // namespace ucxx
