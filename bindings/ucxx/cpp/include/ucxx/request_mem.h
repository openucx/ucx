/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <memory>
#include <string>
#include <utility>

#include <ucp/api/ucp.h>

#include <ucxx/delayed_submission.h>
#include <ucxx/request.h>
#include <ucxx/request_mem_builder.h>
#include <ucxx/typedefs.h>

namespace ucxx {

class Buffer;

namespace internal {
class RecvMemMessage;
}  // namespace internal

/**
 * @brief Send or receive a message with the UCX RMA API.
 *
 * Send or receive a message with the UCX RMA API, using non-blocking UCP calls
 * `ucp_put_nbx` or `ucp_get_nbx`.
 */
class RequestMem : public Request {
 private:
  friend class internal::RecvMemMessage;

  uint64_t _remote_addr{};
  ucp_rkey_h _rkey{};

  /**
   * @brief Private constructor of `ucxx::RequestMem`.
   *
   * This is the internal implementation of `ucxx::RequestMem` constructor, made private not
   * to be called directly. This constructor is made private to ensure all UCXX objects
   * are shared pointers and the correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::Endpoint::memGetBuilder()`
   * - `ucxx::Endpoint::memPutBuilder()`
   * - `ucxx::requestMemBuilder()`
   *
   * @throws ucxx::Error  if `endpoint` is not a valid
   *                      `std::shared_ptr<ucxx::Endpoint>`.
   *
   * @param[in] endpoint            the `std::shared_ptr<Endpoint>` parent component.
   * @param[in] requestData         container of the specified message type, including all
   *                                type-specific data.
   * @param[in] operationName       a human-readable operation name to help identifying
   *                                requests by their types when UCXX logging is enabled.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   * @param[in] callbackFunction    user-defined callback function to call upon completion.
   * @param[in] callbackData        user-defined data to pass to the `callbackFunction`.
   */
  RequestMem(std::shared_ptr<Endpoint> endpoint,
             const std::variant<data::MemPut, data::MemGet> requestData,
             std::string operationName,
             const bool enablePythonFuture                = false,
             RequestCallbackUserFunction callbackFunction = nullptr,
             RequestCallbackUserData callbackData         = nullptr);

  void populateDelayedSubmissionImpl() override;

 public:
  // Allow internal construction without exposing factory functions.
  friend class detail::ConstructorFactory;

  /**
   * @brief Callback executed by UCX when a memory put request is completed.
   *
   * Callback executed by UCX when a memory put request is completed, that will dispatch
   * `ucxx::Request::callback()`.
   *
   * @warning This is not intended to be called by the user, but it currently needs to be
   * a public method so that UCX may access it. In future changes this will be moved to
   * an internal object and remove this method from the public API.
   *
   * @param[in] request the UCX request pointer.
   * @param[in] status  the completion status of the request.
   * @param[in] arg     the pointer to the `ucxx::Request` object that created the
   *                    transfer, effectively `this` pointer as seen by `request()`.
   */
  static void memPutCallback(void* request, ucs_status_t status, void* arg);

  /**
   * @brief Callback executed by UCX when a memory get request is completed.
   *
   * Callback executed by UCX when a memory get request is completed, that will dispatch
   * `ucxx::Request::callback()`.
   *
   * @warning This is not intended to be called by the user, but it currently needs to be
   * a public method so that UCX may access it. In future changes this will be moved to
   * an internal object and remove this method from the public API.
   *
   * @param[in] request the UCX request pointer.
   * @param[in] status  the completion status of the request.
   * @param[in] arg     the pointer to the `ucxx::Request` object that created the
   *                    transfer, effectively `this` pointer as seen by `request()`.
   */
  static void memGetCallback(void* request, ucs_status_t status, void* arg);

  /**
   * @brief Create and submit a memory get or put request.
   *
   * This is the method that should be called to actually submit memory request get or put
   * request. It is meant to be called from `populateDelayedSubmission()`, which is decided
   * at the discretion of `std::shared_ptr<ucxx::Worker>`. See `populateDelayedSubmission()`
   * for more details.
   */
  void request();
};

}  // namespace ucxx
