/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <utility>

#include <ucxx/request_builder_base.h>
#include <ucxx/request_data.h>

namespace ucxx {

// Forward declarations
class Component;
class Request;
class RequestFlush;

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::RequestFlush>` objects.
 *
 * @copydoc ucxx_request_builder_pattern
 *
 * The `endpointOrWorker` and `requestData` are required and must be provided to
 * `requestFlushBuilder()`.
 *
 * @code{.cpp}
 *   auto req = ucxx::requestFlushBuilder(endpointOrWorker, flushData)
 *                .pythonFuture(true)
 *                .callbackFunction(callback)
 *                .callbackData(callbackData)
 *                .build();
 *
 *   std::shared_ptr<ucxx::RequestFlush> flushReq =
 *     ucxx::requestFlushBuilder(endpointOrWorker, flushData);
 * @endcode
 */
class RequestFlushBuilder : public RequestCallbackBuilderBase<RequestFlushBuilder> {
 private:
  std::shared_ptr<Component> _endpointOrWorker;  ///< Parent endpoint or worker (required)
  data::Flush _requestData;                      ///< Request-specific data (required)

 public:
  /**
   * @brief Constructor for `RequestFlushBuilder` with required parameters.
   *
   * @param[in] endpointOrWorker  the parent component, which may be a
   *                              `std::shared_ptr<Endpoint>` or `std::shared_ptr<Worker>`.
   * @param[in] requestData       container of the flush request data.
   */
  explicit RequestFlushBuilder(std::shared_ptr<Component> endpointOrWorker,
                               data::Flush requestData);

  /**
   * @brief Build and return the `RequestFlush`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestFlush>` object.
   */
  [[nodiscard]] std::shared_ptr<RequestFlush> build();

  /**
   * @brief Implicit conversion operator to `shared_ptr<RequestFlush>`.
   *
   * @return The constructed `shared_ptr<ucxx::RequestFlush>` object.
   */
  operator std::shared_ptr<RequestFlush>();

  /**
   * @brief Implicit conversion operator to `shared_ptr<Request>`.
   *
   * @return The constructed request as `shared_ptr<ucxx::Request>`.
   */
  operator std::shared_ptr<Request>();
};

/**
 * @brief Create a RequestFlushBuilder for constructing a `shared_ptr<ucxx::RequestFlush>`.
 *
 * @param[in] endpointOrWorker  the parent component (required).
 * @param[in] requestData       container of the flush request data (required).
 * @return A RequestFlushBuilder object that can be used to set optional parameters.
 */
[[nodiscard]] inline RequestFlushBuilder requestFlushBuilder(
  std::shared_ptr<Component> endpointOrWorker, data::Flush requestData)
{
  return RequestFlushBuilder(std::move(endpointOrWorker), std::move(requestData));
}

}  // namespace ucxx
