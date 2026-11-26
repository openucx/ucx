/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/buffer.h>
#include <ucxx/endpoint.h>
#include <ucxx/future.h>
#include <ucxx/request.h>
#include <ucxx/request_tag_multi_builder.h>

namespace ucxx {

class RequestTagMulti;

/**
 * @brief Container for data required by a `ucxx::RequestTagMulti`.
 *
 * Container for the data required by a `ucxx::RequestTagMulti`, such as the
 * `ucxx::RequestTag` that is doing the operation, as well as buffers to send from or
 * receive at.
 */
struct BufferRequest {
  std::shared_ptr<Request> request{nullptr};  ///< The `ucxx::RequestTag` of a header or frame
  std::shared_ptr<std::string> stringBuffer{nullptr};  ///< Serialized `Header`
  std::shared_ptr<Buffer> buffer{nullptr};  ///< Internally allocated buffer to receive a frame

  BufferRequest();
  ~BufferRequest();

  BufferRequest(const BufferRequest&)            = delete;
  BufferRequest& operator=(BufferRequest const&) = delete;
  BufferRequest(BufferRequest&& o)               = delete;
  BufferRequest& operator=(BufferRequest&& o)    = delete;
};

/**
 * @brief Pre-defined type for a pointer to an `ucxx::BufferRequest`.
 *
 * A pre-defined type for a pointer to a `ucxx::BufferRequest`, used as a convenience type.
 */
typedef std::shared_ptr<BufferRequest> BufferRequestPtr;

/**
 * @brief Send or receive multiple messages with the UCX Tag API.
 *
 * Send or receive multiple messages with the UCX Tag API. This is done combining multiple
 * messages with `ucxx::RequestTag`, first sending/receiving a header, followed by
 * sending/receiving the user messages. Intended primarily for use with Python, such that
 * the program can then only wait for the completion of one future and thus reduce
 * potentially expensive iterations over multiple futures.
 */
class RequestTagMulti : public Request {
 private:
  size_t _totalRequests{0};  ///< The total number of child requests tracked for completion
  std::mutex
    _completedRequestsMutex{};   ///< Mutex to control access to completed requests container
  size_t _completedRequests{0};  ///< Count requests that already completed
  ucs_status_t _finalStatus{
    UCS_OK};  ///< Shortcut to the final status, a.k.a. the first error to occur

 public:
  std::vector<BufferRequestPtr> _bufferRequests{};  ///< Container of all requests posted
  bool _isFilled{false};                            ///< Whether the all requests have been posted

 private:
  RequestTagMulti()                                  = delete;
  RequestTagMulti(const RequestTagMulti&)            = delete;
  RequestTagMulti& operator=(RequestTagMulti const&) = delete;
  RequestTagMulti(RequestTagMulti&& o)               = delete;
  RequestTagMulti& operator=(RequestTagMulti&& o)    = delete;

  /**
   * @brief Protected constructor of a multi-buffer tag receive request.
   *
   * Construct multi-buffer tag receive request, registering the request to the
   * `std::shared_ptr<Endpoint>` parent so that it may be canceled if necessary. This
   * constructor is responsible for creating a Python future that can be later awaited
   * in Python asynchronous code, which is independent of the Python futures used by
   * the underlying `ucxx::RequestTag` object, which will be invisible to the user. Once
   * the initial setup is complete, `callback()` is called to initiate receiving by posting
   * the first request to receive a header.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::Endpoint::tagMultiRecvBuilder()`
   * - `ucxx::Endpoint::tagMultiSendBuilder()`
   * - `ucxx::requestTagMultiBuilder()`
   *
   * @param[in] endpoint            the `std::shared_ptr<Endpoint>` parent component
   * @param[in] requestData         container of the specified message type, including all
   *                                type-specific data.
   * @param[in] operationName       a human-readable operation name to help identifying
   *                                requests by their types when UCXX logging is enabled.
   * @param[in] enablePythonFuture  whether a python future should be created and
   *                                subsequently notified.
   */
  RequestTagMulti(std::shared_ptr<Endpoint> endpoint,
                  const std::variant<data::TagMultiSend, data::TagMultiReceive> requestData,
                  std::string operationName,
                  const bool enablePythonFuture);

  void populateDelayedSubmissionImpl() override;

  /**
   * @brief Receive all frames.
   *
   * Once the header(s) has(have) been received, receiving frames containing the actual data
   * is the next step. This method parses the header(s) and creates as many
   * `ucxx::RequestTag` objects as necessary, each one that will handle a single sending or
   * receiving a single frame.
   *
   * Finally, the object is marked as filled, meaning that all requests were already
   * scheduled and are waiting for completion.
   *
   * @throws std::runtime_error if called by a send request.
   */
  void recvFrames();

  /**
   * @brief Receive a message with header.
   *
   * Create the request to receive a message with header, setting
   * `ucxx::RequestTagMulti::callback` as the user-defined callback of `ucxx::RequestTag` to
   * handle the next request.
   *
   * @throws std::runtime_error if called by a send request.
   */
  void recvHeader();

  /**
   * @brief Send all header(s) and frame(s).
   *
   * Build header request(s) and send them, followed by requests to send all frame(s).
   */
  void send();

 public:
  // Allow internal construction without exposing factory functions.
  friend class detail::ConstructorFactory;

  /**
   * @brief `ucxx::RequestTagMulti` destructor.
   *
   * Free internal resources.
   */
  virtual ~RequestTagMulti();

  /**
   * @brief Mark request as completed.
   *
   * Mark a single `ucxx::RequestTag` as completed. This method is passed as the
   * user-defined callback to the `ucxx::RequestTag` constructor, which will then be
   * executed when that completes.
   *
   * When this method is called, the request that completed will be pushed into a container
   * which will be later used to evaluate if all tracked requests completed and set the
   * final status of the multi-transfer request and the Python future, if enabled. The final
   * status is either `UCS_OK` if all underlying requests completed successfully, otherwise
   * it will contain the status of the first failing request, for granular information the
   * user may still verify each of the underlying requests individually.
   *
   * @param[in] status the status of the request being completed.
   */
  void markCompleted(ucs_status_t status);

  /**
   * @brief Callback to submit request to receive new header or frames.
   *
   * When a receive multi-transfer tag request is created or has received a new header, this
   * callback must be executed to ensure the next request to receive is submitted.
   *
   * If no requests for the present `ucxx::RequestTagMulti` transfer have been posted yet,
   * create one receiving a message with header. If the previous received request is header
   * containing the `next` flag set, then the next request is another header. Otherwise, the
   * next incoming message(s) is(are) frame(s).
   *
   * When called, the callback receives a single argument, the status of the current request.
   *
   * @param[in] status the status of the request being completed.
   * @throws std::runtime_error if called by a send request.
   */
  void recvCallback(ucs_status_t status);

  void cancel() override;
};

/**
 * @brief Pre-defined type for a pointer to an `ucxx::RequestTagMulti`.
 *
 * A pre-defined type for a pointer to a `ucxx::RequestTagMulti`, used as a convenience type.
 */
typedef std::shared_ptr<RequestTagMulti> RequestTagMultiPtr;

}  // namespace ucxx
