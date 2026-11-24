/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <ucp/api/ucp.h>

namespace ucxx {

class Listener;
class Worker;

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::Listener>` objects.
 *
 * This class implements the builder pattern for `std::shared_ptr<ucxx::Listener>`.
 * Construction happens when `build()` is called or when the builder is implicitly converted
 * to `std::shared_ptr<Listener>`.
 */
class ListenerBuilder final {
 public:
  /**
   * @brief Constructor for `ListenerBuilder`.
   *
   * @param[in] worker worker from which to create the listener.
   * @param[in] port port which the listener should bind to.
   * @param[in] callback callback to execute on incoming connections.
   * @param[in] callbackArgs argument to pass to the callback.
   */
  ListenerBuilder(std::shared_ptr<Worker> worker,
                  uint16_t port,
                  ucp_listener_conn_callback_t callback,
                  void* callbackArgs);

  /** @brief `ListenerBuilder` destructor. */
  ~ListenerBuilder();

  /** @brief Copy constructor (deep-copies internal state). */
  ListenerBuilder(const ListenerBuilder& other);
  /** @brief Copy assignment operator (deep-copies internal state). */
  ListenerBuilder& operator=(const ListenerBuilder& other);
  /** @brief Move constructor. */
  ListenerBuilder(ListenerBuilder&&) noexcept;
  /** @brief Move assignment operator. */
  ListenerBuilder& operator=(ListenerBuilder&&) noexcept;

  /**
   * @brief Implicit conversion operator to `shared_ptr<Listener>`.
   *
   * @return The constructed `shared_ptr<ucxx::Listener>` object.
   */
  operator std::shared_ptr<Listener>();

  /**
   * @brief Set the hostname or IP address the listener should bind to.
   *
   * By default the listener binds to all interfaces (wildcard). Use this to restrict
   * the listener to a specific interface, e.g., `"127.0.0.1"` to accept connections
   * from the local host only.
   *
   * @param[in] host hostname or IP address the listener should bind to, an empty
   *                 string binds to all interfaces.
   * @return Reference to this builder for method chaining.
   */
  ListenerBuilder& host(std::string host);

  /**
   * @brief Build and return the `Listener`.
   *
   * Each call to build() creates a new `Listener` instance with the current parameters.
   *
   * @return The constructed `shared_ptr<ucxx::Listener>` object.
   */
  [[nodiscard]] std::shared_ptr<Listener> build();

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

}  // namespace ucxx
