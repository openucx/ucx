/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
class RequestMem;

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::RequestMem>` objects.
 *
 * @copydoc ucxx_request_builder_pattern
 *
 * The `endpoint` and `requestData` are required and must be provided to
 * `requestMemBuilder()`.
 *
 * @code{.cpp}
 *   auto req = ucxx::requestMemBuilder(endpoint, memPutData)
 *                .pythonFuture(true)
 *                .callbackFunction(callback)
 *                .callbackData(callbackData)
 *                .build();
 *
 *   std::shared_ptr<ucxx::RequestMem> memReq =
 *     ucxx::requestMemBuilder(endpoint, memGetData);
 * @endcode
 */
class RequestMemBuilder : public RequestCallbackBuilderBase<RequestMemBuilder> {
 private:
  std::shared_ptr<Endpoint> _endpoint;                    ///< Parent endpoint (required)
  std::variant<data::MemPut, data::MemGet> _requestData;  ///< Request-specific data (required)
  uint64_t _remoteAddressOffset{0};                       ///< Offset from the remote base address

 public:
  /**
   * @brief Constructor for `RequestMemBuilder` with required parameters.
   *
   * @param[in] endpoint     the `std::shared_ptr<Endpoint>` parent component.
   * @param[in] requestData  container of the specified message type, including all
   *                         type-specific data.
   */
  explicit RequestMemBuilder(std::shared_ptr<Endpoint> endpoint,
                             std::variant<data::MemPut, data::MemGet> requestData);

  /**
   * @brief Set the offset from the remote base address.
   *
   * @param[in] offset offset in bytes from the remote base address.
   * @return Reference to this builder for method chaining.
   */
  RequestMemBuilder& remoteAddressOffset(uint64_t offset) &;

  /**
   * @brief Set the offset from the remote base address on a temporary builder.
   *
   * @param[in] offset offset in bytes from the remote base address.
   * @return Rvalue reference to this builder for method chaining.
   */
  RequestMemBuilder&& remoteAddressOffset(uint64_t offset) &&;

  /**
   * @brief Build and return the `RequestMem`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestMem>` object.
   */
  [[nodiscard]] std::shared_ptr<RequestMem> build();

  /**
   * @brief Implicit conversion operator to `shared_ptr<RequestMem>`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestMem>` object.
   */
  operator std::shared_ptr<RequestMem>();

  /**
   * @brief Implicit conversion operator to `shared_ptr<Request>`.
   *
   * @return The constructed request as `shared_ptr<ucxx::Request>`.
   */
  operator std::shared_ptr<Request>();
};

/**
 * @brief Create a RequestMemBuilder for constructing a `shared_ptr<ucxx::RequestMem>`.
 *
 * @param[in] endpoint     the parent endpoint (required).
 * @param[in] requestData  container of the specified message type (required).
 * @return A RequestMemBuilder object that can be used to set optional parameters.
 */
[[nodiscard]] inline RequestMemBuilder requestMemBuilder(
  std::shared_ptr<Endpoint> endpoint, std::variant<data::MemPut, data::MemGet> requestData)
{
  return RequestMemBuilder(std::move(endpoint), std::move(requestData));
}

}  // namespace ucxx
