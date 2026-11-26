/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <netdb.h>

#include <memory>
#include <string>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/address.h>
#include <ucxx/component.h>
#include <ucxx/endpoint_builder.h>
#include <ucxx/exception.h>
#include <ucxx/inflight_requests.h>
#include <ucxx/listener.h>
#include <ucxx/remote_key_builder.h>
#include <ucxx/request.h>
#include <ucxx/request_am_builder.h>
#include <ucxx/request_endpoint_close_builder.h>
#include <ucxx/request_flush_builder.h>
#include <ucxx/request_mem_builder.h>
#include <ucxx/request_stream_builder.h>
#include <ucxx/request_tag_builder.h>
#include <ucxx/request_tag_multi_builder.h>
#include <ucxx/typedefs.h>
#include <ucxx/utils/sockaddr.h>
#include <ucxx/worker.h>

namespace ucxx {

class Component;
class Request;
class RequestEndpointClose;

class RequestEndpointCloseBuilder;

namespace detail {
void registerInflightRequest(std::shared_ptr<Component> component, std::shared_ptr<Request> req);
}  // namespace detail

/**
 * @brief Deleter for a endpoint parameters object.
 *
 * Deleter used during allocation of a `ucp_ep_params_t*` to handle automated deletion of
 * the object when its reference count goes to zero.
 */
struct EpParamsDeleter {
  /**
   * @brief Execute the deletion.
   *
   * Execute the deletion of the `ucp_ep_params_t*` object.
   *
   * param[in] ptr  the point to the object to be deleted.
   */
  void operator()(ucp_ep_params_t* ptr);
};

/**
 * @brief Component encapsulating a UCP endpoint.
 *
 * The UCP layer provides a handle to access endpoints in form of `ucp_ep_h` object,
 * this class encapsulates that object and provides methods to simplify its handling.
 */
class Endpoint : public Component {
 private:
  ucp_ep_h _handle{nullptr};          ///< Handle to the UCP endpoint
  ucp_ep_h _originalHandle{nullptr};  ///< Handle to the UCP endpoint, after it was previously
                                      ///< closed, used for logging purposes only
  bool _endpointErrorHandling{true};  ///< Whether the endpoint enables error handling
  std::unique_ptr<InflightRequests> _inflightRequests{
    std::make_unique<InflightRequests>()};  ///< The inflight requests
  std::mutex _mutex{};  ///< Mutex used during close to prevent race conditions between
                        ///< application thread and `ucxx::Endpoint::setCloseCallback()`
                        ///< that may run asynchronously on another thread.
  ucs_status_t _status{UCS_INPROGRESS};  ///< Endpoint status
  std::atomic<bool> _closing{false};     ///< Prevent calling close multiple concurrent times.
  EndpointCloseCallbackUserFunction _closeCallback{nullptr};  ///< Close callback to call
  EndpointCloseCallbackUserData _closeCallbackArg{
    nullptr};  ///< Argument to be passed to close callback

  /**
   * @brief Private constructor of `ucxx::Endpoint`.
   *
   * This is the internal implementation of `ucxx::Endpoint` constructor, made private not
   * to be called directly. This constructor is made private to ensure all UCXX objects
   * are shared pointers and the correct lifetime management of each one.
   *
   * This constructor does not fully initialize the `ucxx::Endpoint` object, the caller
   * must call the `create()` method immediately after this to complete the construction.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::Listener::endpointBuilder()`
   * - `ucxx::Worker::endpointBuilder()`
   * - `ucxx::endpointBuilder()`
   *
   * @param[in] workerOrListener      the parent component, which may either be a
   *                                  `std::shared_ptr<Listener>` or
   *                                  `std::shared_ptr<Worker>`.
   * @param[in] endpointErrorHandling whether to enable endpoint error handling.
   */
  Endpoint(std::shared_ptr<Component> workerOrListener, bool endpointErrorHandling);

  /**
   * @brief Create the underlying UCP endpoint of `ucxx::Endpoint`.
   *
   * This is the internal implementation of `ucxx::Endpoint` creation. This method completes
   * the initialization with the creation of the UCP endpoint and must be always called
   * after the private constructor is called.
   *
   * @param[in] params                parameters specifying UCP endpoint capabilities.
   */
  void create(ucp_ep_params_t* params);

  /**
   * @brief Register an inflight request.
   *
   * Called each time a new transfer request is made by the `Endpoint`, such that it may
   * be canceled when necessary. Also schedules requests to be canceled immediately after
   * registration if the endpoint error handler has been called with an error.
   *
   * @param[in] request the request to register.
   *
   * @return the request that was registered (i.e., the `request` argument itself).
   */
  [[nodiscard]] std::shared_ptr<Request> registerInflightRequest(std::shared_ptr<Request> request);

  /**
   * @brief Create an endpoint close request while preserving endpoint close state.
   *
   * Marks the endpoint as closing before creating the close request, combines the per-request
   * callback with the endpoint close callback, and registers the request as inflight.
   *
   * @param[in] force               force endpoint close if `true`, flush otherwise.
   * @param[in] enablePythonFuture  whether a python future should be created and subsequently
   *                                notified.
   * @param[in] callbackFunction    user-defined callback function to call upon completion.
   * @param[in] callbackData        user-defined data to pass to the `callbackFunction`.
   *
   * @return Request to be subsequently checked for the completion and its state, or `nullptr`
   *         if the endpoint has already closed or is already in process of closing.
   */
  [[nodiscard]] std::shared_ptr<RequestEndpointClose> closeRequest(
    const bool force,
    const bool enablePythonFuture,
    EndpointCloseCallbackUserFunction callbackFunction,
    EndpointCloseCallbackUserData callbackData);

  /**
   * @brief The error callback registered at endpoint creation time.
   *
   * When the endpoint is created with error handling support this method is registered as
   * the callback to be called when the endpoint is closing, it is responsible for checking
   * the closing status and update internal state accordingly. If error handling support is
   * not active, this method is not registered nor called.
   *
   * The signature for this method must match `ucp_err_handler_cb_t`.
   */
  friend void endpointErrorCallback(void* arg, ucp_ep_h ep, ucs_status_t status);

  /**
   * @brief Allow the endpoint close builder to preserve endpoint close state.
   */
  friend class RequestEndpointCloseBuilder;

  /**
   * @brief Allow request builders to register newly-created requests.
   */
  friend void detail::registerInflightRequest(std::shared_ptr<Component> component,
                                              std::shared_ptr<Request> req);

 public:
  Endpoint()                           = delete;
  Endpoint(const Endpoint&)            = delete;
  Endpoint& operator=(Endpoint const&) = delete;
  Endpoint(Endpoint&& o)               = delete;
  Endpoint& operator=(Endpoint&& o)    = delete;

  ~Endpoint();

  // Allow internal construction without exposing factory functions.
  friend class detail::ConstructorFactory;

  /**
   * @brief Get the underlying `ucp_ep_h` handle.
   *
   * Lifetime of the `ucp_ep_h` handle is managed by the `ucxx::Endpoint` object and its
   * ownership is non-transferrable. Once the `ucxx::Endpoint` is destroyed the handle
   * is not valid anymore, it is the user's responsibility to ensure the owner's lifetime
   * while using the handle.
   *
   * @code{.cpp}
   * // endpoint is `std::shared_ptr<ucxx::Endpoint>`
   * ucp_ep_h endpointHandle = endpoint->getHandle();
   * @endcode
   *
   * @returns The underlying `ucp_ep_h` handle.
   */
  [[nodiscard]] ucp_ep_h getHandle();

  /**
   * @brief Check whether the endpoint is still alive.
   *
   * Check whether the endpoint is still alive, generally `true` until `closeBlocking()` is
   * called, a request submitted by `closeBuilder()` completes or the endpoint errors
   * and the error handling procedure is executed. Always `true` if endpoint error handling
   * is disabled.
   *
   * @returns whether the endpoint is still alive if endpoint enables error handling, always
   *          returns `true` if error handling is disabled.
   */
  [[nodiscard]] bool isAlive() const;

  /**
   * @brief Raises an exception if an error occurred.
   *
   * Raises an exception if an error occurred and error handling is enabled for the
   * endpoint, no-op otherwise.
   *
   * @throws ucxx::ConnectionResetError if `UCP_ERR_CONNECTION_RESET` occurred.
   * @throws ucxx::Error                if any other UCP error occurred.
   */
  void raiseOnError();

  /**
   * @brief Remove reference to request from internal container.
   *
   * Remove the reference to a specific request from the internal container. This should
   * be called when a request has completed and the `ucxx::Endpoint` does not need to keep
   * track of it anymore.
   *
   * @param[in] request shared pointer to the request
   */
  void removeInflightRequest(std::shared_ptr<Request> request);

  /**
   * @brief Cancel inflight requests.
   *
   * Cancel inflight requests, returning the total number of requests that were scheduled
   * for cancelation. After the requests are scheduled for cancelation, the caller must
   * progress the worker and check the result of `getCancelingSize()`, all requests are only
   * canceled when `getCancelingSize()` returns `0`.
   *
   * @returns Number of requests that were scheduled for cancelation.
   */
  size_t cancelInflightRequests();

  /**
   * @brief Check the number of inflight requests being canceled.
   *
   * Check the number of inflight requests that were scheduled for cancelation with
   * `cancelInflightRequests()` who have not yet completed cancelation. To ensure their
   * cancelation is completed, the worker must be progressed until this method returns `0`.
   *
   * @returns Number of requests that are in process of cancelation.
   */
  [[nodiscard]] size_t getCancelingSize() const;

  /**
   * @brief Cancel inflight requests.
   *
   * Cancel inflight requests and block until all requests complete cancelation, returning
   * the total number of requests that were canceled.  This is usually executed by
   * `closeBlocking()`, when pending requests will no longer be able to complete.
   *
   * If the parent worker is running a progress thread, a maximum timeout may be specified
   * for which the close operation will wait. This can be particularly important for cases
   * where the progress thread might be attempting to acquire a resource (e.g., the Python
   * GIL) while the current thread owns that resource. In particular for Python, the
   * `~Endpoint()` will call this method for which we can't release the GIL when the garbage
   * collector runs and destroys the object.
   *
   * @param[in] period      maximum period to wait for a generic pre/post progress thread
   *                        operation will wait for.
   * @param[in] maxAttempts maximum number of attempts to close endpoint, only applicable
   *                         if worker is running a progress thread and `period > 0`.
   *
   * @returns Number of requests that were canceled.
   */
  size_t cancelInflightRequestsBlocking(uint64_t period = 0, uint64_t maxAttempts = 1);

  /**
   * @brief Register a user-defined callback to call when endpoint closes.
   *
   * Register a user-defined callback and argument that is later called immediately after
   * the endpoint closes. The callback is executed either if the endpoint closed
   * successfully after completing and disconnecting from the remote endpoint, but more
   * importantly when any error occurs, allowing the application to be notified immediately
   * after such an event occurred.
   *
   * @throws  std::runtime_error  if the endpoint is closing or has already closed and this
   *                              is not removing the close callback (setting both
   *                              `closeCallback` and `closeCallbackArg` to `nullptr`)
   *
   * @param[in] closeCallback     `std::function` to a function definition return `void` and
   *                              receiving a single opaque pointer.
   * @param[in] closeCallbackArg  pointer to optional user-allocated callback argument.
   */
  void setCloseCallback(EndpointCloseCallbackUserFunction closeCallback,
                        EndpointCloseCallbackUserData closeCallbackArg);

  /**
   * @brief Create a builder for an active message send operation.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion submits the request.
   *
   * @param[in] buffer                  a raw pointer to the data to be sent.
   * @param[in] length                  the size in bytes of the message to be sent.
   * @param[in] memoryType              the memory type of the buffer.
   *
   * @returns Builder to configure optional parameters and submit the request.
   */
  [[nodiscard]] RequestAmBuilder amSendBuilder(const void* const buffer,
                                               const size_t length,
                                               const ucs_memory_type_t memoryType);

  /**
   * @brief Create a builder for an active message send operation with explicit policy parameters.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion submits the request.
   *
   * @param[in] buffer              a raw pointer to the data to be sent.
   * @param[in] length              the size in bytes of the message to be sent.
   * @param[in] params              active message send parameters.
   *
   * @returns Builder to configure optional parameters and submit the request.
   */
  [[nodiscard]] RequestAmBuilder amSendBuilder(const void* const buffer,
                                               const size_t length,
                                               const AmSendParams& params);

  /**
   * @brief Create a builder for an active message send operation with IOV datatype.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion submits the request.
   *
   * @param[in] iov                 vector of IOV segments to be sent.
   * @param[in] params              active message send parameters. Datatype must be
   *                                `UCP_DATATYPE_IOV`.
   *
   * @returns Builder to configure optional parameters and submit the request.
   */
  [[nodiscard]] RequestAmBuilder amSendBuilder(std::vector<ucp_dt_iov_t> iov,
                                               const AmSendParams& params);

  /**
   * @brief Create a builder for an active message receive operation.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion submits the request.
   *
   * @returns Builder to configure optional parameters and submit the request.
   */
  [[nodiscard]] RequestAmBuilder amRecvBuilder();

  /**
   * @brief Create a builder for a memory put operation.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion submits the request.
   *
   * @param[in] buffer              a raw pointer to the data to be sent.
   * @param[in] length              the size in bytes of the message to be sent.
   * @param[in] remoteAddr          the destination remote memory address to write to.
   * @param[in] rkey                the remote memory key associated with the remote memory
   *                                address.
   *
   * @returns Builder to configure optional parameters and submit the request.
   */
  [[nodiscard]] RequestMemBuilder memPutBuilder(const void* const buffer,
                                                size_t length,
                                                uint64_t remoteAddr,
                                                ucp_rkey_h rkey);

  /**
   * @brief Create a builder for a memory put operation.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion submits the request.
   *
   * @param[in] buffer              a raw pointer to the data to be sent.
   * @param[in] length              the size in bytes of the message to be sent.
   * @param[in] remoteKey           the remote memory key associated with the remote memory
   *                                address.
   *
   * @returns Builder to configure optional parameters and submit the request.
   */
  [[nodiscard]] RequestMemBuilder memPutBuilder(const void* const buffer,
                                                size_t length,
                                                std::shared_ptr<ucxx::RemoteKey> remoteKey);

  /**
   * @brief Create a builder for a memory get operation.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion submits the request.
   *
   * @param[in] buffer              a raw pointer to the data to be received.
   * @param[in] length              the size in bytes of the message to be received.
   * @param[in] remoteAddr          the source remote memory address to read from.
   * @param[in] rkey                the remote memory key associated with the remote memory
   *                                address.
   *
   * @returns Builder to configure optional parameters and submit the request.
   */
  [[nodiscard]] RequestMemBuilder memGetBuilder(void* buffer,
                                                size_t length,
                                                uint64_t remoteAddr,
                                                ucp_rkey_h rkey);

  /**
   * @brief Create a builder for a memory get operation.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion submits the request.
   *
   * @param[in] buffer              a raw pointer to the data to be received.
   * @param[in] length              the size in bytes of the message to be received.
   * @param[in] remoteKey           the remote memory key associated with the remote memory
   *                                address.
   *
   * @returns Builder to configure optional parameters and submit the request.
   */
  [[nodiscard]] RequestMemBuilder memGetBuilder(void* buffer,
                                                size_t length,
                                                std::shared_ptr<ucxx::RemoteKey> remoteKey);

  /**
   * @brief Create a builder for a stream send operation.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion submits the request.
   *
   * @param[in] buffer              a raw pointer to the data to be sent.
   * @param[in] length              the size in bytes of the message to be sent.
   *
   * @returns Builder to configure optional parameters and submit the request.
   */
  [[nodiscard]] RequestStreamBuilder streamSendBuilder(const void* const buffer, size_t length);

  /**
   * @brief Create a builder for a stream receive operation.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion submits the request.
   *
   * @param[in] buffer              a raw pointer to pre-allocated memory where resulting
   *                                data will be stored.
   * @param[in] length              the size in bytes of the message to be received.
   *
   * @returns Builder to configure optional parameters and submit the request.
   */
  [[nodiscard]] RequestStreamBuilder streamRecvBuilder(void* buffer, size_t length);

  /**
   * @brief Create a builder for a tag send operation.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion submits the request.
   * Completion makes the local buffer safe to reuse, but does not confirm that the peer
   * received the message. Applications requiring peer delivery before a force-close must use
   * an application-level acknowledgement.
   *
   * @param[in] buffer              a raw pointer to the data to be sent.
   * @param[in] length              the size in bytes of the message to be sent.
   * @param[in] tag                 the tag to match.
   *
   * @returns Builder to configure optional parameters and submit the request.
   */
  [[nodiscard]] RequestTagBuilder tagSendBuilder(const void* const buffer, size_t length, Tag tag);

  /**
   * @brief Create a builder for a tag receive operation.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion submits the request.
   *
   * @param[in] buffer              a raw pointer to pre-allocated memory where resulting
   *                                data will be stored.
   * @param[in] length              the size in bytes of the message to be received.
   * @param[in] tag                 the tag to match.
   * @param[in] tagMask             the tag mask to use.
   *
   * @returns Builder to configure optional parameters and submit the request.
   */
  [[nodiscard]] RequestTagBuilder tagRecvBuilder(void* buffer,
                                                 size_t length,
                                                 Tag tag,
                                                 TagMask tagMask);

  /**
   * @brief Create a builder for a multi-buffer tag send operation.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion submits the request.
   * Completion makes the local buffers safe to reuse, but does not confirm that the peer
   * received every frame. Applications requiring peer delivery before a force-close must use
   * an application-level acknowledgement.
   *
   * @param[in] buffer              a vector of raw pointers to the data frames to be sent.
   * @param[in] size                a vector of size in bytes of each frame to be sent.
   * @param[in] isCUDA              a vector indicating whether each frame is CUDA memory.
   * @param[in] tag                 the tag to match.
   *
   * @returns Builder to configure optional parameters and submit the request.
   */
  [[nodiscard]] RequestTagMultiBuilder tagMultiSendBuilder(const std::vector<const void*>& buffer,
                                                           const std::vector<size_t>& size,
                                                           const std::vector<int>& isCUDA,
                                                           const Tag tag);

  /**
   * @brief Create a builder for a multi-buffer tag receive operation.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion submits the request.
   *
   * @param[in] tag                 the tag to match.
   * @param[in] tagMask             the tag mask to use.
   *
   * @returns Builder to configure optional parameters and submit the request.
   */
  [[nodiscard]] RequestTagMultiBuilder tagMultiRecvBuilder(const Tag tag, const TagMask tagMask);

  /**
   * @brief Create a builder for a flush operation.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion submits the request.
   *
   * @returns Builder to configure optional parameters and submit the request.
   */
  [[nodiscard]] RequestFlushBuilder flushBuilder();

  /**
   * @brief Get `ucxx::Worker` component from a worker or listener object.
   *
   * A `std::shared_ptr<ucxx::Endpoint>` needs to be created and registered by
   * `std::shared_ptr<ucxx::Worker>`, but the endpoint may be a child of a
   * `std::shared_ptr<ucxx::Listener>` object. For convenience, this method can be used to
   * get the `std::shared_ptr<ucxx::Worker>` which the endpoint is associated with.
   *
   * @returns The `std::shared_ptr<ucxx::Worker>` which the endpoint is associated with.
   */
  [[nodiscard]] std::shared_ptr<Worker> getWorker();

  /**
   * @brief Create a builder for a remote key from serialized data.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion unpacks the remote key using this endpoint.
   *
   * @param[in] serializedRemoteKey serialized remote key data.
   * @returns Builder to create the remote key.
   */
  [[nodiscard]] RemoteKeyBuilder remoteKeyBuilder(SerializedRemoteKey serializedRemoteKey);

  /**
   * @brief Create a builder for a non-blocking endpoint close operation.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion submits the request.
   *
   * @returns Builder to configure optional parameters and submit the close request.
   */
  [[nodiscard]] RequestEndpointCloseBuilder closeBuilder();

  /**
   * @brief Close the endpoint while keeping the object alive.
   *
   * Close the endpoint without requiring to destroy the object, first canceling inflight
   * requests and then waiting for the close operation to complete when possible. This may
   * be useful when `std::shared_ptr<ucxx::Request>` objects are still alive.
   *
   * If the endpoint was created with error handling support, the error callback will be
   * executed, implying the user-defined callback will also be executed if one was
   * registered with `setCloseCallback()`.
   *
   * If the parent worker is running a progress thread, `period` and `maxAttempts` bound how
   * long this method waits for progress-thread submission and completion checks. If close
   * completion cannot be observed before all attempts expire, the endpoint status is set to
   * `UCS_ERR_ENDPOINT_TIMEOUT` before returning. Any UCP endpoint close request handle
   * returned by `ucp_ep_close_nbx()` is released before this method returns.
   *
   * @param[in] period      maximum period to wait for a generic pre/post progress thread
   *                        operation will wait for.
   * @param[in] maxAttempts maximum number of attempts to close endpoint, only applicable
   *                        if worker is running a progress thread and `period > 0`.
   */
  void closeBlocking(uint64_t period = 0, uint64_t maxAttempts = 1);
};

}  // namespace ucxx
