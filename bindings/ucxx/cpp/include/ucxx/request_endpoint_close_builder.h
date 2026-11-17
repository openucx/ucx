/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <utility>

#include <ucxx/request_builder_base.h>
#include <ucxx/request_data.h>

namespace ucxx {

// Forward declarations
class Endpoint;
class Request;
class RequestEndpointClose;

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::RequestEndpointClose>` objects.
 *
 * @copydoc ucxx_request_builder_pattern
 *
 * The `endpoint` and `requestData` are required and must be provided to
 * `requestEndpointCloseBuilder()`.
 * Building the request preserves `ucxx::Endpoint::closeBuilder()` lifecycle semantics:
 * only one close request may be submitted for an endpoint.
 *
 * @code{.cpp}
 *   auto req = ucxx::requestEndpointCloseBuilder(endpoint, closeData)
 *                .pythonFuture(true)
 *                .callbackFunction(callback)
 *                .callbackData(callbackData)
 *                .build();
 *
 *   std::shared_ptr<ucxx::RequestEndpointClose> closeReq =
 *     ucxx::requestEndpointCloseBuilder(endpoint, closeData);
 * @endcode
 */
class RequestEndpointCloseBuilder : public RequestCallbackBuilderBase<RequestEndpointCloseBuilder> {
 private:
  std::shared_ptr<Endpoint> _endpoint;  ///< Parent endpoint (required)
  data::EndpointClose _requestData;     ///< Request-specific data (required)

 public:
  /**
   * @brief Constructor for `RequestEndpointCloseBuilder` with required parameters.
   *
   * @param[in] endpoint     the `std::shared_ptr<Endpoint>` parent component.
   * @param[in] requestData  container of the endpoint close request data.
   */
  explicit RequestEndpointCloseBuilder(std::shared_ptr<Endpoint> endpoint,
                                       data::EndpointClose requestData);

  /**
   * @brief Build and return the `RequestEndpointClose`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestEndpointClose>` object, or `nullptr`
   *         if the endpoint has already closed or is already in process of closing.
   */
  [[nodiscard]] std::shared_ptr<RequestEndpointClose> build();

  /**
   * @brief Implicit conversion operator to `shared_ptr<RequestEndpointClose>`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestEndpointClose>` object, or `nullptr`
   *         if the endpoint has already closed or is already in process of closing.
   */
  operator std::shared_ptr<RequestEndpointClose>();

  /**
   * @brief Implicit conversion operator to `shared_ptr<Request>`.
   *
   * @return The constructed request as `shared_ptr<ucxx::Request>`, or `nullptr`
   *         if the endpoint has already closed or is already in process of closing.
   */
  operator std::shared_ptr<Request>();
};

/**
 * @brief Create a RequestEndpointCloseBuilder for constructing a
 * `shared_ptr<ucxx::RequestEndpointClose>`.
 *
 * @param[in] endpoint     the parent endpoint (required).
 * @param[in] requestData  container of the endpoint close request data (required).
 * @return A RequestEndpointCloseBuilder object that can be used to set optional parameters.
 */
[[nodiscard]] inline RequestEndpointCloseBuilder requestEndpointCloseBuilder(
  std::shared_ptr<Endpoint> endpoint, data::EndpointClose requestData)
{
  return RequestEndpointCloseBuilder(std::move(endpoint), std::move(requestData));
}

}  // namespace ucxx
