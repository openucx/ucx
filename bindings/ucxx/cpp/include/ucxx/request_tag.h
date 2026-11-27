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
#include <ucxx/request_tag_builder.h>
#include <ucxx/typedefs.h>

namespace ucxx {

/**
 * @brief Send or receive a message with the UCX Tag API.
 *
 * Send or receive a message with the UCX Tag API, using non-blocking UCP calls
 * `ucp_tag_send_nbx` or `ucp_tag_recv_nbx`.
 */
class RequestTag : public Request {
 private:
  bool _cancelRequested{false};  ///< Whether to cancel immediately after delayed submission.

  /**
   * @brief Private constructor of `ucxx::RequestTag`.
   *
   * This is the internal implementation of `ucxx::RequestTag` constructor, made private not
   * to be called directly. This constructor is made private to ensure all UCXX objects
   * are shared pointers and the correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::Endpoint::tagRecvBuilder()`
   * - `ucxx::Endpoint::tagSendBuilder()`
   * - `ucxx::Worker::tagRecvBuilder()`
   * - `ucxx::requestTagBuilder()`
   *
   * @throws ucxx::Error  if send is `true` and `endpointOrWorker` is not a
   *                      `std::shared_ptr<ucxx::Endpoint>`.
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
  RequestTag(
    std::shared_ptr<Component> endpointOrWorker,
    const std::variant<data::TagSend, data::TagReceive, data::TagReceiveWithHandle> requestData,
    std::string operationName,
    const bool enablePythonFuture                = false,
    RequestCallbackUserFunction callbackFunction = nullptr,
    RequestCallbackUserData callbackData         = nullptr);

  void populateDelayedSubmissionImpl() override;

 public:
  // Allow internal construction without exposing factory functions.
  friend class detail::ConstructorFactory;

  /**
   * @brief Cancel the tag request.
   *
   * A receive created from a probed message handle must consume that handle before it can
   * be canceled. If delayed submission has not occurred yet, defer cancelation until the
   * progress context has submitted and processed the receive.
   */
  void cancel() override;

  /**
   * @brief Create and submit a tag request.
   *
   * This is the method that should be called to actually submit a tag request. It is meant
   * to be called from `populateDelayedSubmission()`, which is decided at the discretion of
   * `std::shared_ptr<ucxx::Worker>`. See `populateDelayedSubmission()` for more details.
   */
  void request();

  /**
   * @brief Callback executed by UCX when a tag send request is completed.
   *
   * Callback executed by UCX when a tag send request is completed, that will dispatch
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
  static void tagSendCallback(void* request, ucs_status_t status, void* arg);

  /**
   * @brief Callback executed by UCX when a tag receive request is completed.
   *
   * Callback executed by UCX when a tag receive request is completed, that will dispatch
   * `ucxx::RequestTag::callback()`.
   *
   * @warning This is not intended to be called by the user, but it currently needs to be
   * a public method so that UCX may access it. In future changes this will be moved to
   * an internal object and remove this method from the public API.
   *
   * @param[in] request the UCX request pointer.
   * @param[in] status  the completion status of the request.
   * @param[in] info    information of the completed transfer provided by UCX, includes
   *                    length of message received used to verify for truncation.
   * @param[in] arg     the pointer to the `ucxx::Request` object that created the
   *                    transfer, effectively `this` pointer as seen by `request()`.
   */
  static void tagRecvCallback(void* request,
                              ucs_status_t status,
                              const ucp_tag_recv_info_t* info,
                              void* arg);

  /**
   * @brief Implementation of the tag receive request callback.
   *
   * Implementation of the tag receive request callback. Verify whether the message was
   * truncated and set that state if necessary, and finally dispatch
   * `ucxx::Request::callback()`.
   *
   * @warning This is not intended to be called by the user, but it currently needs to be
   * a public method so that UCX may access it. In future changes this will be moved to
   * an internal object and remove this method from the public API.
   *
   * @param[in] request the UCX request pointer.
   * @param[in] status  the completion status of the request.
   * @param[in] info    information of the completed transfer provided by UCX, includes
   *                    length of message received used to verify for truncation.
   */
  void callback(void* request, ucs_status_t status, const ucp_tag_recv_info_t* info);
};

}  // namespace ucxx
