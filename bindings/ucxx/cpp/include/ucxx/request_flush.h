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
#include <ucxx/request_flush_builder.h>
#include <ucxx/typedefs.h>

namespace ucxx {

/**
 * @brief Flush a UCP endpoint or worker.
 *
 * Flush outstanding AMO (Atomic Memory Operation) and RMA (Remote Memory Access) operations
 * on a UCP endpoint or worker.
 */
class RequestFlush : public Request {
 private:
  /**
   * @brief Private constructor of `ucxx::RequestFlush`.
   *
   * This is the internal implementation of `ucxx::RequestFlush` constructor, made private
   * not to be called directly. This constructor is made private to ensure all UCXX objects
   * are shared pointers and the correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::Endpoint::flushBuilder()`
   * - `ucxx::Worker::flushBuilder()`
   * - `ucxx::requestFlushBuilder()`
   *
   * @throws ucxx::Error  if `endpointOrWorker` is not a valid
   *                      `std::shared_ptr<ucxx::Endpoint>` or
   *                      `std::shared_ptr<ucxx::Worker>`.
   *
   * @param[in] endpointOrWorker    the parent component, which may either be a
   *                                `std::shared_ptr<Endpoint>` or
   *                                `std::shared_ptr<Worker>`.
   * @param[in] requestData         container of the specified message type, including all
   *                                type-specific data.
   * @param[in] operationName       a human-readable operation name to help identifying
   *                                requests by their types when UCXX logging is enabled.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   * @param[in] callbackFunction    user-defined callback function to call upon completion.
   * @param[in] callbackData        user-defined data to pass to the `callbackFunction`.
   */
  RequestFlush(std::shared_ptr<Component> endpointOrWorker,
               const data::Flush requestData,
               std::string operationName,
               const bool enablePythonFuture                = false,
               RequestCallbackUserFunction callbackFunction = nullptr,
               RequestCallbackUserData callbackData         = nullptr);

  void populateDelayedSubmissionImpl() override;

 public:
  // Allow internal construction without exposing factory functions.
  friend class detail::ConstructorFactory;

  /**
   * @brief Create and submit a flush request.
   *
   * This is the method that should be called to actually submit a flush request. It is
   * meant to be called from `populateDelayedSubmission()`, which is decided at the
   * discretion of `std::shared_ptr<ucxx::Worker>`. See `populateDelayedSubmission()` for
   * more details.
   */
  void request();

  /**
   * @brief Callback executed by UCX when a flush request is completed.
   *
   * Callback executed by UCX when a flush request is completed, that will dispatch
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
  static void flushCallback(void* request, ucs_status_t status, void* arg);
};

}  // namespace ucxx
