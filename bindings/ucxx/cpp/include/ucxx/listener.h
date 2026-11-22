/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/endpoint_builder.h>
#include <ucxx/listener_builder.h>
#include <ucxx/worker.h>

namespace ucxx {

class Listener;
/**
 * @brief Component encapsulating a UCP listener.
 *
 * The UCP layer provides a handle to access listeners in form of `ucp_listener_h` object,
 * this class encapsulates that object and provides methods to simplify its handling.
 */
class Listener : public Component {
 private:
  ucp_listener_h _handle{nullptr};  ///< The UCP listener handle
  std::string _ip{};                ///< The IP address to which the listener is bound to
  uint16_t _port{0};                ///< The port to which the listener is bound to

  /**
   * @brief Private constructor of `ucxx::Listener`.
   *
   * This is the internal implementation of `ucxx::Listener` constructor, made private not
   * to be called directly.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::Worker::listenerBuilder()`
   * - `ucxx::ListenerBuilder`
   *
   *
   * @param[in] worker        the worker from which to create the listener.
   * @param[in] host          the hostname or IP address which the listener should be
   *                          bound to, an empty string binds to all interfaces
   *                          (wildcard).
   * @param[in] port          the port which the listener should be bound to.
   * @param[in] callback      user-defined callback to be executed on incoming client
   *                          connections.
   * @param[in] callbackArgs  argument to be passed to the callback.
   */
  Listener(std::shared_ptr<Worker> worker,
           std::string host,
           uint16_t port,
           ucp_listener_conn_callback_t callback,
           void* callbackArgs);

 public:
  Listener()                           = delete;
  Listener(const Listener&)            = delete;
  Listener& operator=(Listener const&) = delete;
  Listener(Listener&& o)               = delete;
  Listener& operator=(Listener&& o)    = delete;

  ~Listener();

  // Allow internal construction without exposing factory functions.
  friend class detail::ConstructorFactory;

  /**
   * @brief Create a builder for an endpoint from a connection request.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion creates the endpoint from this listener.
   *
   * @param[in] connRequest handle to connection request delivered by a listener callback.
   * @returns Builder to configure optional endpoint parameters.
   */
  [[nodiscard]] EndpointBuilder endpointBuilder(ucp_conn_request_h connRequest);

  /**
   * @brief Get the underlying `ucp_listener_h` handle.
   *
   * Lifetime of the `ucp_listener_h` handle is managed by the `ucxx::Listener` object and
   * its ownership is non-transferrable. Once the `ucxx::Listener` is destroyed the handle
   * is not valid anymore, it is the user's responsibility to ensure the owner's lifetime
   * while using the handle.
   *
   * @code{.cpp}
   * // listener is `std::shared_ptr<ucxx::Listener>`
   * ucp_listener_h listenerHandle = listener->getHandle();
   * @endcode
   *
   * @returns The underlying `ucp_listener_h` handle.
   */
  [[nodiscard]] ucp_listener_h getHandle();

  /**
   * @brief Get the port to which the listener is bound to.
   *
   * Get the port to which the listener is bound to.
   *
   * @returns the port to which the listener is bound to.
   */
  [[nodiscard]] uint16_t getPort();

  /**
   * @brief Get the IP address to which the listener is bound to.
   *
   * Get the IP address to which the listener is bound to.
   *
   * @returns the IP address to which the listener is bound to.
   */
  [[nodiscard]] std::string getIp();
};

}  // namespace ucxx
