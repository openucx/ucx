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
class Component;
class Request;
class RequestTag;

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::RequestTag>` objects.
 *
 * @copydoc ucxx_request_builder_pattern
 *
 * The `endpointOrWorker` and `requestData` are required and must be provided to
 * `requestTagBuilder()`.
 *
 * @code{.cpp}
 *   auto req = ucxx::requestTagBuilder(endpointOrWorker, tagSendData)
 *                .pythonFuture(true)
 *                .callbackFunction(callback)
 *                .callbackData(callbackData)
 *                .build();
 *
 *   std::shared_ptr<ucxx::RequestTag> tagReq =
 *     ucxx::requestTagBuilder(endpointOrWorker, tagReceiveData);
 * @endcode
 */
class RequestTagBuilder : public RequestCallbackBuilderBase<RequestTagBuilder> {
 private:
  std::shared_ptr<Component> _endpointOrWorker;  ///< Parent endpoint or worker (required)
  std::variant<data::TagSend, data::TagReceive, data::TagReceiveWithHandle>
    _requestData;  ///< Request-specific data (required)

 public:
  /**
   * @brief Constructor for `RequestTagBuilder` with required parameters.
   *
   * @param[in] endpointOrWorker  the parent component, which may be a
   *                              `std::shared_ptr<Endpoint>` or `std::shared_ptr<Worker>`.
   * @param[in] requestData       container of the specified message type, including all
   *                              type-specific data.
   */
  explicit RequestTagBuilder(
    std::shared_ptr<Component> endpointOrWorker,
    std::variant<data::TagSend, data::TagReceive, data::TagReceiveWithHandle> requestData);

  /**
   * @brief Build and return the `RequestTag`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestTag>` object.
   */
  [[nodiscard]] std::shared_ptr<RequestTag> build();

  /**
   * @brief Implicit conversion operator to `shared_ptr<RequestTag>`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestTag>` object.
   */
  operator std::shared_ptr<RequestTag>();

  /**
   * @brief Implicit conversion operator to `shared_ptr<Request>`.
   *
   * @return The constructed request as `shared_ptr<ucxx::Request>`.
   */
  operator std::shared_ptr<Request>();
};

/**
 * @brief Create a RequestTagBuilder for constructing a `shared_ptr<ucxx::RequestTag>`.
 *
 * @param[in] endpointOrWorker  the parent component (required).
 * @param[in] requestData       container of the specified message type (required).
 * @return A RequestTagBuilder object that can be used to set optional parameters.
 */
[[nodiscard]] inline RequestTagBuilder requestTagBuilder(
  std::shared_ptr<Component> endpointOrWorker,
  std::variant<data::TagSend, data::TagReceive, data::TagReceiveWithHandle> requestData)
{
  return RequestTagBuilder(std::move(endpointOrWorker), std::move(requestData));
}

}  // namespace ucxx
