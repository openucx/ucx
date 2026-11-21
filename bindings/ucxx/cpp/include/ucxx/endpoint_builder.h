/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <ucp/api/ucp.h>

namespace ucxx {

class Address;
class Endpoint;
class Listener;
class Worker;

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::Endpoint>` objects.
 *
 * This class implements the builder pattern for `std::shared_ptr<ucxx::Endpoint>`.
 * Construction happens when `build()` is called or when the builder is implicitly converted
 * to `std::shared_ptr<Endpoint>`.
 */
class EndpointBuilder final {
 public:
  /**
   * @brief Constructor for `EndpointBuilder` from hostname and port.
   *
   * @param[in] worker parent worker from which to create the endpoint.
   * @param[in] ipAddress hostname or IP address the listener is bound to.
   * @param[in] port port the listener is bound to.
   */
  EndpointBuilder(std::shared_ptr<Worker> worker, std::string ipAddress, uint16_t port);

  /**
   * @brief Constructor for `EndpointBuilder` from a connection request.
   *
   * @param[in] listener listener from which to create the endpoint.
   * @param[in] connRequest connection request delivered by a listener callback.
   */
  EndpointBuilder(std::shared_ptr<Listener> listener, ucp_conn_request_h connRequest);

  /**
   * @brief Constructor for `EndpointBuilder` from a worker address.
   *
   * @param[in] worker parent worker from which to create the endpoint.
   * @param[in] address address of the remote UCX worker.
   */
  EndpointBuilder(std::shared_ptr<Worker> worker, std::shared_ptr<Address> address);

  /** @brief `EndpointBuilder` destructor. */
  ~EndpointBuilder();

  /** @brief Copy constructor (deep-copies internal state). */
  EndpointBuilder(const EndpointBuilder& other);
  /** @brief Copy assignment operator (deep-copies internal state). */
  EndpointBuilder& operator=(const EndpointBuilder& other);
  /** @brief Move constructor. */
  EndpointBuilder(EndpointBuilder&&) noexcept;
  /** @brief Move assignment operator. */
  EndpointBuilder& operator=(EndpointBuilder&&) noexcept;

  /**
   * @brief Implicit conversion operator to `shared_ptr<Endpoint>`.
   *
   * @return The constructed `shared_ptr<ucxx::Endpoint>` object.
   */
  operator std::shared_ptr<Endpoint>();

  /**
   * @brief Configure endpoint error handling.
   *
   * @param[in] enable whether endpoint error handling is enabled (default: true).
   * @return Reference to this builder for method chaining.
   */
  EndpointBuilder& endpointErrorHandling(bool enable = true);

  /**
   * @brief Build and return the `Endpoint`.
   *
   * Each call to build() creates a new `Endpoint` instance with the current parameters.
   *
   * @return The constructed `shared_ptr<ucxx::Endpoint>` object.
   */
  [[nodiscard]] std::shared_ptr<Endpoint> build();

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

/**
 * @brief Create an EndpointBuilder from hostname and port.
 *
 * @param[in] worker parent worker from which to create the endpoint.
 * @param[in] ipAddress hostname or IP address the listener is bound to.
 * @param[in] port port the listener is bound to.
 * @return An EndpointBuilder object.
 */
[[nodiscard]] inline EndpointBuilder endpointBuilder(std::shared_ptr<Worker> worker,
                                                     std::string ipAddress,
                                                     uint16_t port)
{
  return EndpointBuilder(std::move(worker), std::move(ipAddress), port);
}

/**
 * @brief Create an EndpointBuilder from a connection request.
 *
 * @param[in] listener listener from which to create the endpoint.
 * @param[in] connRequest connection request delivered by a listener callback.
 * @return An EndpointBuilder object.
 */
[[nodiscard]] inline EndpointBuilder endpointBuilder(std::shared_ptr<Listener> listener,
                                                     ucp_conn_request_h connRequest)
{
  return EndpointBuilder(std::move(listener), connRequest);
}

/**
 * @brief Create an EndpointBuilder from a worker address.
 *
 * @param[in] worker parent worker from which to create the endpoint.
 * @param[in] address address of the remote UCX worker.
 * @return An EndpointBuilder object.
 */
[[nodiscard]] inline EndpointBuilder endpointBuilder(std::shared_ptr<Worker> worker,
                                                     std::shared_ptr<Address> address)
{
  return EndpointBuilder(std::move(worker), std::move(address));
}

}  // namespace ucxx
