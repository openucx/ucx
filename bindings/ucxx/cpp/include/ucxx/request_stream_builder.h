/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <utility>
#include <variant>

#include <ucxx/request_builder_base.h>
#include <ucxx/request_data.h>

namespace ucxx {

// Forward declarations
class Endpoint;
class Request;
class RequestStream;

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::RequestStream>` objects.
 *
 * @copydoc ucxx_request_builder_pattern
 *
 * The `endpoint` and `requestData` are required and must be provided to
 * `requestStreamBuilder()`. The `pythonFuture()` method is optional.
 *
 * @code{.cpp}
 *   auto req = ucxx::requestStreamBuilder(endpoint, streamSendData)
 *                .pythonFuture(true)
 *                .build();
 *
 *   std::shared_ptr<ucxx::RequestStream> streamReq =
 *     ucxx::requestStreamBuilder(endpoint, streamReceiveData);
 * @endcode
 */
class RequestStreamBuilder : public RequestBuilderBase<RequestStreamBuilder> {
 private:
  std::shared_ptr<Endpoint> _endpoint;  ///< Parent endpoint (required)
  std::variant<data::StreamSend, data::StreamReceive>
    _requestData;  ///< Request-specific data (required)

 public:
  /**
   * @brief Constructor for `RequestStreamBuilder` with required parameters.
   *
   * @param[in] endpoint     the `std::shared_ptr<Endpoint>` parent component.
   * @param[in] requestData  container of the specified message type, including all
   *                         type-specific data.
   */
  explicit RequestStreamBuilder(std::shared_ptr<Endpoint> endpoint,
                                std::variant<data::StreamSend, data::StreamReceive> requestData);

  /**
   * @brief Build and return the `RequestStream`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestStream>` object.
   */
  [[nodiscard]] std::shared_ptr<RequestStream> build();

  /**
   * @brief Implicit conversion operator to `shared_ptr<RequestStream>`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestStream>` object.
   */
  operator std::shared_ptr<RequestStream>();

  /**
   * @brief Implicit conversion operator to `shared_ptr<Request>`.
   *
   * @return The constructed request as `shared_ptr<ucxx::Request>`.
   */
  operator std::shared_ptr<Request>();
};

/**
 * @brief Create a RequestStreamBuilder for constructing a `shared_ptr<ucxx::RequestStream>`.
 *
 * @param[in] endpoint     the parent endpoint (required).
 * @param[in] requestData  container of the specified message type (required).
 * @return A RequestStreamBuilder object that can be used to set optional parameters.
 */
[[nodiscard]] inline RequestStreamBuilder requestStreamBuilder(
  std::shared_ptr<Endpoint> endpoint,
  std::variant<data::StreamSend, data::StreamReceive> requestData)
{
  return RequestStreamBuilder(std::move(endpoint), std::move(requestData));
}

}  // namespace ucxx
