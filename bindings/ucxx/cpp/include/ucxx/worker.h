/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <variant>

#include <ucp/api/ucp.h>

#include <ucxx/address_builder.h>
#include <ucxx/buffer.h>
#include <ucxx/component.h>
#include <ucxx/context.h>
#include <ucxx/delayed_submission.h>
#include <ucxx/endpoint_builder.h>
#include <ucxx/future.h>
#include <ucxx/inflight_requests.h>
#include <ucxx/listener_builder.h>
#include <ucxx/notifier.h>
#include <ucxx/request_flush_builder.h>
#include <ucxx/request_tag_builder.h>
#include <ucxx/typedefs.h>
#include <ucxx/worker_builder.h>
#include <ucxx/worker_progress_thread.h>

namespace ucxx {

class Component;
class Request;

class WorkerBuilder;

namespace detail {
void registerInflightRequest(std::shared_ptr<Component> component, std::shared_ptr<Request> req);
}  // namespace detail

class Address;
class Buffer;
class Endpoint;
class Listener;
class RequestAm;

namespace internal {
class AmData;
}  // namespace internal

/**
 * @brief Component encapsulating a UCP worker.
 *
 * The UCP layer provides a handle to access workers in form of `ucp_worker_h` object,
 * this class encapsulates that object and provides methods to simplify its handling.
 */
class Worker : public Component {
 private:
  ucp_worker_h _handle{nullptr};        ///< The UCP worker handle
  int _epollFileDescriptor{-1};         ///< The epoll file descriptor
  int _workerFileDescriptor{-1};        ///< The worker file descriptor
  std::mutex _inflightRequestsMutex{};  ///< Mutex to access the inflight requests pool
  std::unique_ptr<InflightRequests> _inflightRequests{
    std::make_unique<InflightRequests>()};  ///< The inflight requests
  std::mutex
    _inflightRequestsToCancelMutex{};  ///< Mutex to access the inflight requests to cancel pool
  std::unique_ptr<InflightRequests> _inflightRequestsToCancel{
    std::make_unique<InflightRequests>()};  ///< The inflight requests scheduled to be canceled
  WorkerProgressThread _progressThread{};   ///< The progress thread object
  std::atomic<bool> _progressThreadRunning{false};      ///< Whether the progress thread is running
  std::atomic<bool> _progressThreadPollingMode{false};  ///< Whether the progress thread polls
  std::atomic<std::thread::id> _progressThreadId{std::thread::id()};  ///< The progress thread ID
  std::function<void(void*)> _progressThreadStartCallback{
    nullptr};  ///< The callback function to execute at progress thread start
  void* _progressThreadStartCallbackArg{
    nullptr};  ///< The argument to be passed to the progress thread start callback
  std::shared_ptr<DelayedSubmissionCollection> _delayedSubmissionCollection{
    nullptr};  ///< Collection of enqueued delayed submissions

  friend class detail::ConstructorFactory;

 protected:
  bool _enableFuture{
    false};  ///< Boolean identifying whether the worker was created with future capability
  bool _enableRequestAttributes{
    false};  ///< Whether request attributes (e.g. UCP debug info) are queried for each request
  std::mutex _futuresPoolMutex{};  ///< Mutex to access the futures pool
  std::queue<std::shared_ptr<Future>>
    _futuresPool{};  ///< Futures pool to prevent running out of fresh futures
  std::shared_ptr<Notifier> _notifier{nullptr};  ///< Notifier object
  std::shared_ptr<internal::AmData>
    _amData;  ///< Worker data made available to Active Messages callback
  BufferType _cudaBufferType{BufferType::Invalid};  ///< Preferred buffer type for CUDA allocations

 private:
  /**
   * @brief Drain the worker for uncaught tag messages received.
   *
   * Called by the destructor, any uncaught tag messages received will be drained so as
   * not to generate UCX warnings.
   */
  void drainWorkerTagRecv();

  /**
   * @brief Get active message receive request.
   *
   * Returns an active message request from the pool if the worker has already begun
   * handling a request with the active messages callback, otherwise creates a new request
   * that is later populated with status and buffer by the active messages callback.
   *
   * @param[in] ep  the endpoint handle where receiving the message, the same handle that
   *                will later be used to reply to the message.
   * @param[in] createAmRecvRequestFunction function to create a new request if one is not
   *                                        already available in the pool.
   *
   * @returns Request to be subsequently checked for the completion state and data.
   */
  [[nodiscard]] std::shared_ptr<RequestAm> getAmRecv(
    ucp_ep_h ep, std::function<std::shared_ptr<RequestAm>()> createAmRecvRequestFunction);

  /**
   * @brief Stop the progress thread if running without raising warnings.
   *
   * Called by the destructor, will stop the progress thread if running without
   * raising warnings.
   */
  void stopProgressThreadNoWarn();

  /**
   * @brief Register an inflight request.
   *
   * Called each time a new transfer request is made by the `Worker`, such that it may
   * be canceled when necessary.
   *
   * @param[in] request the request to register.
   *
   * @return the request that was registered (i.e., the `request` argument itself).
   */
  [[nodiscard]] std::shared_ptr<Request> registerInflightRequest(std::shared_ptr<Request> request);

  /**
   * @brief Progress the worker until all communication events are completed.
   *
   * Iteratively calls `progressOnce()` until all communication events are completed.
   *
   * @returns whether any communication events have been progressed.
   */
  bool progressPending();

  /**
   * @brief Set the preferred buffer type for CUDA allocations.
   *
   * Configure which buffer type to use when allocating CUDA buffers for incoming
   * multi-buffer tag receives.
   *
   * @param[in] bufferType  the preferred buffer type (currently only `BufferType::CCCL`
   *                        is supported).
   *
   * @throws std::invalid_argument if bufferType is not CCCL.
   */
  void setCudaBufferType(BufferType bufferType);

 protected:
  /**
   * @brief Protected constructor of `ucxx::Worker`.
   *
   * This is the internal implementation of `ucxx::Worker` constructor, made protected not
   * to be called directly.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::Context::workerBuilder()`
   * - `ucxx::workerBuilder()`
   *
   *
   * @param[in] context the context from which to create the worker.
   * @param[in] enableDelayedSubmission if `true`, each `ucxx::Request` will not be
   *                                    submitted immediately, but instead delayed to
   *                                    the progress thread. Requires use of the
   *                                    progress thread.
   * @param[in] enableFuture if `true`, notifies the future associated with each
   *                         `ucxx::Request`, currently used only by `ucxx::python::Worker`.
   */
  explicit Worker(std::shared_ptr<Context> context,
                  const bool enableDelayedSubmission = false,
                  const bool enableFuture            = false);

 public:
  Worker()                         = delete;
  Worker(const Worker&)            = delete;
  Worker& operator=(Worker const&) = delete;
  Worker(Worker&& o)               = delete;
  Worker& operator=(Worker&& o)    = delete;

  /**
   * @brief Allow WorkerBuilder to access protected/private constructor.
   */
  friend class WorkerBuilder;

  /**
   * @brief Allow request builders to register newly-created requests.
   */
  friend void detail::registerInflightRequest(std::shared_ptr<Component> component,
                                              std::shared_ptr<Request> req);

  /**
   * @brief `ucxx::Worker` destructor.
   */
  virtual ~Worker();

  /**
   * @brief Get the underlying `ucp_worker_h` handle.
   *
   * Lifetime of the `ucp_worker_h` handle is managed by the `ucxx::Worker` object and
   * its ownership is non-transferrable. Once the `ucxx::Worker` is destroyed the handle
   * is not valid anymore, it is the user's responsibility to ensure the owner's lifetime
   * while using the handle.
   *
   * @code{.cpp}
   * // worker is `std::shared_ptr<ucxx::Worker>`
   * ucp_worker_h workerHandle = worker->getHandle();
   * @endcode
   *
   * @returns The underlying `ucp_worker_h` handle.
   */
  [[nodiscard]] ucp_worker_h getHandle();

  /**
   * @brief Get information about the underlying `ucp_worker_h` object.
   *
   * Convenience wrapper for `ucp_worker_print_info()` to get information about the
   * underlying UCP worker handle and return it as a string.
   *
   * @returns String containing information about the UCP worker.
   */
  [[nodiscard]] std::string getInfo();

  /**
   * @brief Initialize blocking progress mode.
   *
   * Initialize blocking progress mode, creates internal file descriptors to handle blocking
   * progress by waiting for the UCP worker to notify the file descriptors. This method is
   * supposed to be called when usage of `progressWorkerEvent()` is intended, before the
   * first call to `progressWorkerEvent()`. If using polling mode only via
   * `progress()`/`progressOnce()` calls or wake-up with `waitProgress()`, this method should
   * not be called.
   *
   * In blocking mode, the user should call `progressWorkerEvent()` to block and then progress
   * the worker as new events arrive. `wakeProgressEvent()` may be called to forcefully wake
   * this method, for example to shutdown the application.
   *
   * @code{.cpp}
   * // worker is `std::shared_ptr<ucxx::Worker>`
   *
   * // Block until UCX's wakes up for an incoming event, then fully progresses the
   * // worker
   * worker->initBlockingProgressMode();
   * worker->progressWorkerEvent();
   *
   * // All events have been progressed.
   * @endcode
   *
   * @throws std::ios_base::failure if creating any of the file descriptors or setting their
   *                                statuses.
   */
  void initBlockingProgressMode();

  /**
   * @brief Get the epoll file descriptor associated with the worker.
   *
   * Get the epoll file descriptor associated with the worker when running in blocking mode.
   * The worker only has an associated epoll file descriptor after
   * `initBlockingProgressMode()` is executed.
   *
   * The file descriptor is destroyed as part of the `ucxx::Worker` destructor, thus any
   * reference to it shall not be used after that.
   *
   * @throws std::runtime_error if `initBlockingProgressMode()` was not executed to run the
   *                            worker in blocking progress mode.
   *
   * @returns the file descriptor.
   */
  int getEpollFileDescriptor();

  /**
   * @brief Arm the UCP worker.
   *
   * Wrapper for `ucp_worker_arm`, checking its return status for errors and raising an
   * exception if an error occurred.
   *
   * @throws ucxx::Error if an error occurred while attempting to arm the worker.
   *
   * @returns `true` if worker was armed successfully, `false` if its status was `UCS_ERR_BUSY`.
   */
  bool arm();

  /**
   * @brief Progress worker event while in blocking progress mode.
   *
   * Blocks until a new worker event has happened and the worker notifies the file descriptor
   * associated with it, or `epollTimeout` has elapsed. Requires blocking progress mode to
   * be initialized with `initBlockingProgressMode()` before the first call to this method.
   * Additionally ensure inflight messages pending for cancelation are canceled.
   *
   * @code{.cpp}
   * // worker is `std::shared_ptr<ucxx::Worker>`
   *
   * // Block until UCX's wakes up for an incoming event, then fully progresses the
   * // worker
   * worker->initBlockingProgressMode();
   * worker->progressWorkerEvent();
   *
   * // All events have been progressed.
   * @endcode
   *
   * @param[in] epollTimeout  timeout in ms when waiting for worker event, or -1 to block
   *                          indefinitely.
   *
   * @throws std::ios_base::failure if creating any of the file descriptors or setting their
   *                                statuses.
   *
   * @returns `true` if any communication was progressed, `false` otherwise.
   */
  bool progressWorkerEvent(const int epollTimeout = -1);

  /**
   * @brief Signal the worker that an event happened.
   *
   * Signals that an event has happened while, causing both either `progressWorkerEvent()`
   * or `waitProgress()` to immediately wake-up.
   *
   * @code{.cpp}
   * // worker is `std::shared_ptr<ucxx::Worker>`
   *
   * void progressThread() {
   *   // Block until UCX's wakes up for an incoming event, then fully progresses the
   *   // worker.
   *   worker->initBlockingProgressMode();
   *   worker->progressWorkerEvent();
   *
   *   // Will reach this point and exit after 3 seconds
   * }
   *
   * void otherThread() {
   *   // Signals the worker after 3 seconds
   *   std::this_thread::sleep_for(std::chrono::seconds(3));
   *   worker->signal();
   * }
   *
   * void mainThread() {
   *   t1 = std::thread(progressThread);
   *   t2 = std::thread(otherThread);
   *   t1.join();
   *   t2.join();
   * }
   * @endcode
   *
   * @throws ucxx::Error if an error occurred while attempting to signal the worker.
   */
  void signal();

  /**
   * @brief Block until an event has happened, then progresses.
   *
   * Blocks until an event has happened as part of UCX's wake-up mechanism and progress
   * the worker. Additionally ensure inflight messages pending for cancelation are canceled.
   *
   * @code{.cpp}
   * // worker is `std::shared_ptr<ucxx::Worker>`
   *
   * // Block until UCX's wakes up for an incoming event, then fully progresses the
   * // worker
   * worker->waitProgress();
   * worker->progress();
   *
   * // All events have been progressed.
   * @endcode
   *
   * @throws ucxx::Error if an error occurred while attempting to arm the worker.
   *
   * @returns `true` if any communication was progressed, `false` otherwise.
   */
  bool waitProgress();

  /**
   * @brief Progress the worker only once.
   *
   * Wrapper for `ucp_worker_progress`.
   *
   * @code{.cpp}
   * // worker is `std::shared_ptr<ucxx::Worker>`
   * while (!worker->progressOnce()) ;
   *
   * // All events have been progressed.
   * @endcode
   *
   * @returns `true` if any communication was progressed, `false` otherwise.
   */
  bool progressOnce();

  /**
   * @brief Progress the worker until all communication events are completed.
   *
   * Iteratively calls `progressOnce()` until all communication events are completed.
   * Additionally ensure inflight messages pending for cancelation are canceled.
   *
   * @code{.cpp}
   * // worker is `std::shared_ptr<ucxx::Worker>`
   * worker->progress();
   *
   * // All events have been progressed and inflight pending for cancelation were canceled.
   * @endcode
   *
   * @returns whether any communication events have been progressed.
   */
  bool progress();

  /**
   * @brief Register delayed request submission.
   *
   * Register `ucxx::Request` for delayed submission. When the `ucxx::Worker` is created
   * with `enableDelayedSubmission=true`, calling actual UCX transfer routines will not
   * happen immediately and instead will be submitted later by the worker thread.
   *
   * The purpose of this method is to offload as much as possible any work to the worker
   * thread, thus decreasing computation on the caller thread, but potentially increasing
   * transfer latency.
   *
   * @param[in] request  the request to which the callback belongs, ensuring it remains
   *                     alive until the callback is invoked.
   * @param[in] callback the callback set to execute the UCP transfer routine during the
   *                     worker thread loop.
   */
  void registerDelayedSubmission(std::shared_ptr<Request> request,
                                 DelayedSubmissionCallbackType callback);

  /**
   * @brief Register callback to be executed in progress thread before progressing.
   *
   * Register callback to be executed in the current or next iteration of the progress
   * thread before the worker is progressed. There is no guarantee that the callback will
   * be executed in the current or next iteration, this depends on where the progress thread
   * is in the current iteration when this callback is registered.
   *
   * The purpose of this method is to schedule operations to be executed in the progress
   * thread, such as endpoint creation and closing, so that progressing doesn't ever need
   * to occur in the application thread when using a progress thread.
   *
   * If `period` is `0` this is a blocking call that only returns when the callback has been
   * executed and will always return `true`, and if `period` is a positive integer the time
   * in nanoseconds will be waited for the callback to complete and return `true` in the
   * successful case or `false` otherwise. However, if the callback is not cancelable
   * anymore (i.e., it has already started), this method will keep retrying and may never
   * return if the callback never completes, it is unsafe to return as this would allow the
   * caller to destroy the callback and its resources causing undefined behavior. `period`
   * only applies if the worker progress thread is running, otherwise the callback is
   * immediately executed.
   *
   * @param[in] callback  the callback to execute before progressing the worker.
   * @param[in] period    the time in nanoseconds to wait for the callback to complete.
   *
   * @returns `true` if the callback was successfully executed or `false` if timed out.
   */
  [[nodiscard]] bool registerGenericPre(DelayedSubmissionCallbackType callback,
                                        uint64_t period = 0);

  /**
   * @brief Register callback to be executed in progress thread before progressing.
   *
   * Register callback to be executed in the current or next iteration of the progress
   * thread after the worker is progressed. There is no guarantee that the callback will
   * be executed in the current or next iteration, this depends on where the progress thread
   * is in the current iteration when this callback is registered.
   *
   * The purpose of this method is to schedule operations to be executed in the progress
   * thread, immediately after progressing the worker completes.
   *
   * If `period` is `0` this is a blocking call that only returns when the callback has been
   * executed and will always return `true`, and if `period` is a positive integer the time
   * in nanoseconds will be waited for the callback to complete and return `true` in the
   * successful case or `false` otherwise. However, if the callback is not cancelable
   * anymore (i.e., it has already started), this method will keep retrying and may never
   * return if the callback never completes, it is unsafe to return as this would allow the
   * caller to destroy the callback and its resources causing undefined behavior. `period`
   * only applies if the worker progress thread is running, otherwise the callback is
   * immediately executed.
   *
   * @param[in] callback  the callback to execute before progressing the worker.
   * @param[in] period    the time in nanoseconds to wait for the callback to complete.
   *
   * @returns `true` if the callback was successfully executed or `false` if timed out.
   */
  [[nodiscard]] bool registerGenericPost(DelayedSubmissionCallbackType callback,
                                         uint64_t period = 0);

  /**
   * @brief Inquire if worker has been created with delayed submission enabled.
   *
   * Check whether the worker has been created with delayed submission enabled.
   *
   * @returns `true` if delayed submission is enabled, `false` otherwise.
   */
  [[nodiscard]] bool isDelayedRequestSubmissionEnabled() const;

  /**
   * @brief Inquire if worker has been created with future support.
   *
   * Check whether the worker has been created with future support.
   *
   * @returns `true` if future support is enabled, `false` otherwise.
   */
  [[nodiscard]] bool isFutureEnabled() const;

  /**
   * @brief Inquire if worker has been created with request attributes querying enabled.
   *
   * Check whether the worker has been created with request attributes querying enabled.
   * When enabled, each `ucxx::Request` will have its UCP attributes (such as the debug
   * string) queried immediately after submission, making them available via
   * `ucxx::Request::queryAttributes()`. Querying request attributes has a
   * non-negligible runtime cost and is therefore disabled by default.
   *
   * @returns `true` if request attributes querying is enabled, `false` otherwise.
   */
  [[nodiscard]] bool isRequestAttributesEnabled() const noexcept;

  /**
   * @brief Get the preferred buffer type for CUDA allocations.
   *
   * Returns the buffer type used when allocating CUDA buffers for incoming
   * multi-buffer tag receives. Defaults to CCCL if compiled with CCCL support,
   * otherwise Invalid.
   *
   * @returns The preferred `BufferType` for CUDA allocations.
   */
  [[nodiscard]] BufferType getCudaBufferType() const;

  /**
   * @brief Populate the futures pool.
   *
   * To avoid taking blocking resources (such as the Python GIL) for every new future
   * required by each `ucxx::Request`, the `ucxx::Worker` maintains a pool of futures
   * that can be acquired when a new `ucxx::Request` is created. Currently the pool has
   * a maximum size of 100 objects, and will refill once it goes under 50, otherwise
   * calling this functions results in a no-op.
   *
   * @throws std::runtime_error if future support is not implemented.
   */
  virtual void populateFuturesPool();

  /**
   * @brief Clear the futures pool.
   *
   * Clear the futures pool, ensuring all references are removed and thus avoiding
   * reference cycles that prevent the `ucxx::Worker` and other resources from cleaning
   * up on time.
   *
   * @throws std::runtime_error if future support is not implemented.
   */
  virtual void clearFuturesPool();

  /**
   * @brief Get a future from the pool.
   *
   * Get a future from the pool. If the pool is empty,
   * `ucxx::Worker::populateFuturesPool()` is called and a warning is raised, since
   * that likely means the user is missing to call the aforementioned method regularly.
   *
   * @throws std::runtime_error if future support is not implemented.
   *
   * @returns The `shared_ptr<ucxx::python::Future>` object
   */
  [[nodiscard]] virtual std::shared_ptr<Future> getFuture();

  /**
   * @brief Block until a request event.
   *
   * Blocks until some communication is completed and future is ready to be notified,
   * shutdown was initiated or a timeout occurred (only if `periodNs > 0`). This method is
   * intended for use from the notifier (such as the Python thread running it), where that
   * thread will block until one of the aforementioned events occur.
   *
   * @throws std::runtime_error if future support is not implemented.
   *
   * @returns `RequestNotifierWaitState::Ready` if some communication completed,
   *          `RequestNotifierWaitStats::Timeout` if a timeout occurred, or
   *          `RequestNotifierWaitStats::Shutdown` if shutdown has initiated.
   */
  [[nodiscard]] virtual RequestNotifierWaitState waitRequestNotifier(uint64_t periodNs);

  /**
   * @brief Notify futures of each completed communication request.
   *
   * Notifies futures of each completed communication request of their new status. This
   * method is intended to be used from the Notifier (such as the Python thread running it),
   * where the thread will call `waitRequestNotifier()` and block until some communication
   * is completed, and then call this method to notify all futures. If this is notifying
   * a Python future, the thread where this method is called from must be using the same
   * Python event loop as the thread that submitted the transfer request.
   *
   * @throws std::runtime_error if future support is not implemented.
   */
  virtual void runRequestNotifier();

  /**
   * @brief Signal the notifier to terminate.
   *
   * Signals the notifier to terminate, awakening the `waitRequestNotifier()` blocking call.
   *
   * @throws std::runtime_error if future support is not implemented.
   */
  virtual void stopRequestNotifierThread();

  /**
   * @brief Set callback to be executed at the progress thread start.
   *
   * Sets a callback that will be executed at the beginning of the progress thread. This
   * can be used to initialize any resources that are required to be available on the thread
   * the worker will be progressed from, such as a CUDA context.
   *
   * @param[in] callback function to execute during progress thread start
   * @param[in] callbackArg argument to be passed to the callback function
   */
  void setProgressThreadStartCallback(std::function<void(void*)> callback, void* callbackArg);

  /**
   * @brief Start the progress thread.
   *
   * Spawns a new thread that will take care of continuously progressing the worker. The
   * thread can progress the worker in blocking mode, using `progressWorkerEvent()` only
   * when worker events happen, or in polling mode by continuously calling `progress()`
   * (incurs in high CPU utilization).
   *
   * @param[in] pollingMode   use polling mode if `true`, or blocking mode if `false`.
   * @param[in] epollTimeout  timeout in ms when waiting for worker event, or -1 to block
   *                          indefinitely, only applicable if `pollingMode==true`.
   */
  void startProgressThread(const bool pollingMode = false, const int epollTimeout = 1);

  /**
   * @brief Stop the progress thread.
   *
   * Stop the progress thread.
   *
   * May be called by the user at any time, and also called during destructor if the
   * worker thread was ever started.
   */
  void stopProgressThread();

  /**
   * @brief Inquire if worker has a progress thread running.
   *
   * Check whether the worker currently has a progress thread running.
   *
   * @returns `true` if a progress thread is running, `false` otherwise.
   */
  [[nodiscard]] bool isProgressThreadRunning();

  /**
   * @brief Get the progress thread ID.
   *
   * Get the progress thread ID, only valid if `startProgressThread()` was called.
   *
   * @returns the progress thread ID.
   */
  [[nodiscard]] std::thread::id getProgressThreadId();

  /**
   * @brief Cancel inflight requests.
   *
   * Cancel inflight requests, returning the total number of requests that were canceled.
   * This is usually executed during the progress loop.
   *
   * If the parent worker is running a progress thread, a maximum timeout may be specified
   * for which the close operation will wait. This can be particularly important for cases
   * where the progress thread might be attempting to acquire a resource (e.g., the Python
   * GIL) while the current thread owns that resource. In particular for Python, the
   * `~Worker()` will call this method for which we can't release the GIL when the garbage
   * collector runs and destroys the object.
   *
   * @param[in] period      maximum period to wait for a generic pre/post progress thread
   *                        operation will wait for.
   * @param[in] maxAttempts maximum number of attempts to close endpoint, only applicable
   *                         if worker is running a progress thread and `period > 0`.
   *
   * @returns Number of requests that were canceled.
   */
  size_t cancelInflightRequests(uint64_t period = 0, uint64_t maxAttempts = 1);

  /**
   * @brief Schedule cancelation of inflight requests.
   *
   * Schedule inflight request to be canceled when `cancelInflightRequests()` is executed
   * the next time, usually during the progress loop. This is usually called from a
   * `ucxx::Endpoint`, for example when the error callback was called, signaling that
   * inflight requests for that endpoint will not be completed successfully and should be
   * canceled.
   *
   * @param[in] trackedRequests the requests tracked by a child of this class to be
   *                            scheduled for cancelation.
   */
  void scheduleRequestCancel(TrackedRequests trackedRequests);

  /**
   * @brief Remove reference to request from internal container.
   *
   * Remove the reference to a specific request from the internal container. This should
   * be called when a request has completed and the `ucxx::Worker` does not need to keep
   * track of it anymore.
   *
   * @param[in] request shared pointer to the request
   */
  void removeInflightRequest(std::shared_ptr<Request> request);

  /**
   * @brief Check for uncaught tag messages.
   *
   * Checks the worker for any uncaught tag messages. An uncaught tag message is any
   * tag message that has been fully or partially received by the worker, but not matched
   * by a corresponding `ucp_tag_recv_*` call. Additionally, returns information about the
   * tag message.
   *
   * When `remove` is `false` (default), the message remains in the receive queue and
   * a normal tag receive operation must be posted to consume it. When `remove` is `true`,
   * the message is removed from the queue and a message handle is returned that can be
   * passed directly to `tagRecvWithHandleBuilder` for efficient consumption, in this case
   * the caller is required to build a `tagRecvWithHandleBuilder` request, otherwise a
   * warning will be thrown during destruction of the returned object.
   *
   * Note this is a non-blocking call, if this is being used to actively check for an
   * incoming message the worker should be constantly progress until a valid probe is
   * returned.
   *
   * @code{.cpp}
   * // `worker` is `std::shared_ptr<ucxx::Worker>`
   * auto probe = worker->tagProbe(0);
   * assert(!probe->isMatched());
   *
   * // `ep` is a remote `std::shared_ptr<ucxx::Endpoint` to the local `worker`
   * ep->tagSendBuilder(buffer, length, 0).build();
   *
   * probe = worker->tagProbe(0);
   * assert(probe->isMatched());
   * assert(probe->getInfo().tag == 0);
   * assert(probe->getInfo().length == length);
   * @endcode
   *
   * @param[in] tag      the tag to match.
   * @param[in] tagMask  the tag mask to use.
   * @param[in] remove   if `true`, remove the message from the queue and return a message
   *                     handle; if `false`, leave the message in the queue.
   *
   * @returns TagProbeInfo containing whether a message was matched, the tag receive information
   *          (when matched=true), and optionally the message handle for efficient reception
   *          (when matched=true and remove=true).
   */
  [[nodiscard]] std::shared_ptr<TagProbeInfo> tagProbe(const Tag tag,
                                                       const TagMask tagMask = TagMaskFull,
                                                       const bool remove     = false) const;

  /**
   * @brief Create a builder for a tag receive operation.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion submits the request.
   *
   * @param[in] buffer            a raw pointer to pre-allocated memory where resulting
   *                              data will be stored.
   * @param[in] length            the size in bytes of the message to be received.
   * @param[in] tag               the tag to match.
   * @param[in] tagMask           the tag mask to use.
   *
   * @returns Builder to configure optional parameters and submit the request.
   */
  [[nodiscard]] RequestTagBuilder tagRecvBuilder(void* buffer,
                                                 size_t length,
                                                 Tag tag,
                                                 TagMask tagMask);

  /**
   * @brief Create a builder for a tag receive operation using a message handle.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion submits the request.
   *
   * @param[in] buffer            a raw pointer to pre-allocated memory where resulting
   *                              data will be stored.
   * @param[in] probeInfo         the TagProbeInfo object containing message length and handle.
   *
   * @returns Builder to configure optional parameters and submit the request.
   */
  [[nodiscard]] RequestTagBuilder tagRecvWithHandleBuilder(void* buffer,
                                                           std::shared_ptr<TagProbeInfo> probeInfo);

  /**
   * @brief Create a builder for a tag receive operation using a message handle.
   *
   * If the message does not fit in the receive buffer, the resulting request completes
   * with `UCS_ERR_MESSAGE_TRUNCATED`. Calling `Request::checkError()` then raises
   * `ucxx::MessageTruncatedError`.
   *
   * @param[in] buffer     a raw pointer to pre-allocated receive memory.
   * @param[in] length     the size in bytes of the receive buffer.
   * @param[in] probeInfo  the TagProbeInfo object containing message length and handle.
   *
   * @returns Builder to configure optional parameters and submit the request.
   */
  [[nodiscard]] RequestTagBuilder tagRecvWithHandleBuilder(void* buffer,
                                                           size_t length,
                                                           std::shared_ptr<TagProbeInfo> probeInfo);

  /**
   * @brief Create a builder for this worker's address.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion gets the address of the underlying UCX worker object.
   *
   * @returns Builder to create the worker address.
   */
  [[nodiscard]] AddressBuilder addressBuilder();

  /**
   * @brief Create a builder for an endpoint to a worker listening on an IP and port.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion creates the endpoint from this worker.
   *
   * @param[in] ipAddress string containing the IP address of the remote worker.
   * @param[in] port port number where the remote worker is listening at.
   * @returns Builder to configure optional endpoint parameters.
   */
  [[nodiscard]] EndpointBuilder endpointBuilder(std::string ipAddress, uint16_t port);

  /**
   * @brief Create a builder for an endpoint to a worker located at a UCX address.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion creates the endpoint from this worker.
   *
   * @param[in] address address of the remote UCX worker.
   * @returns Builder to configure optional endpoint parameters.
   */
  [[nodiscard]] EndpointBuilder endpointBuilder(std::shared_ptr<Address> address);

  /**
   * @brief Create a builder for a listener on this worker.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion starts a listener on this worker.
   *
   * @param[in] port port number where to listen at.
   * @param[in] callback callback to handle each incoming connection.
   * @param[in] callbackArgs pointer to argument to pass to the callback.
   * @returns Builder to create the listener.
   */
  [[nodiscard]] ListenerBuilder listenerBuilder(uint16_t port,
                                                ucp_listener_conn_callback_t callback,
                                                void* callbackArgs);

  /**
   * @brief Register allocator for active messages.
   *
   * Register a new allocator for active messages. By default, only one allocator is defined
   * for host memory (`UCS_MEMORY_TYPE_HOST`), and is used as a fallback when an allocator
   * for the source's memory type is unavailable. In many circumstances relying exclusively
   * on the host allocator is undesirable, for example when transferring CUDA buffers the
   * destination is always going to be a host buffer and prevent the use of transports such
   * as NVLink or InfiniBand+GPUDirectRDMA. For that reason it's important that the user
   * defines those allocators that are important for the application.
   *
   * If the `memoryType` has already been registered, the previous allocator will be
   * replaced by the new one. Be careful when doing this after transfers have started, there
   * are no guarantees that inflight messages have not already been allocated with the old
   * allocator for that type.
   *
   * @code{.cpp}
   * // context is `std::shared_ptr<ucxx::Context>`
   * auto worker = context->workerBuilder().build();
   *
   * worker->registerAmAllocator(UCS_MEMORY_TYPE_CUDA, [](size_t length) {
   *   return std::make_shared<ucxx::CCCLBuffer>(length);
   * });
   * @endcode
   *
   * @param[in] memoryType  the memory type the allocator will be used for.
   * @param[in] allocator   the allocator callable that will be used to allocate new
   *                        active message buffers.
   */
  void registerAmAllocator(ucs_memory_type_t memoryType, AmAllocatorType allocator);

  /**
   * @brief Register receiver callback for active messages.
   *
   * Register a new receiver callback for active messages. By default, active messages do
   * not execute any callbacks on the receiving end unless one is specified when sending
   * the message. If the message sender specifies a callback receiver identifier then the
   * remote receiver needs to have a callback registered with the same identifier to
   * execute when the request completes. To ensure multiple applications that do not know
   * about each other can have coexisting callbacks where receiver identifiers may have
   * the same value, an owner must be specified as well, which has the form of a string and
   * should be reasonably unique to prevent accidentally calling callbacks from a separate
   * application, thus names like "A" or "UCX" are discouraged in favor of more descriptive
   * names such as "MyFastCommsProject", and the name "ucxx" is reserved.
   *
   * Because it is impossible to predict which callback would be called in such an event,
   * the registered callback cannot be changed, thus calling this method with the same
   * given owner and identifier will throw `std::runtime_error`.
   *
   * @code{.cpp}
   * // `worker` is `std::shared_ptr<ucxx::Worker>`
   * auto callback = [](std::shared_ptr<ucxx::Request> req) {
   *   std::cout << "The UCXX request address is " << (void*)req.get() << std::endl;
   * };
   *
   * worker->registerAmReceiverCallback({"MyFastApp", 0}, callback};
   * @endcode
   *
   * @throws std::runtime_error if a callback with same given owner and identifier is
   *                            already registered, or if the reserved owner name "ucxx"
   *                            is specified.
   *
   * @param[in] info      the owner name and unique identifier of the receiver callback.
   * @param[in] callback  the callback to execute when the active message is received.
   */
  void registerAmReceiverCallback(AmReceiverCallbackInfo info, AmReceiverCallbackType callback);

  /**
   * @brief Check for uncaught active messages.
   *
   * Checks the worker for any uncaught active messages. An uncaught active message is any
   * active message that has been fully or partially received by the worker, but not matched
   * by a corresponding active-message receive request.
   *
   * @code{.cpp}
   * // `worker` is `std::shared_ptr<ucxx::Worker>`
   * // `ep` is a remote `std::shared_ptr<ucxx::Endpoint` to the local `worker`
   * assert(!worker->amProbe(ep->getHandle()));
   *
   * ep->amSendBuilder(buffer, length, UCS_MEMORY_TYPE_HOST).build();
   *
   * assert(worker->amProbe(0));
   * @endcode
   *
   * @returns `true` if any uncaught messages were received, `false` otherwise.
   */
  [[nodiscard]] bool amProbe(const ucp_ep_h endpointHandle) const;

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
   * @brief Worker attributes reported by `ucp_worker_query`.
   */
  struct Attributes {
    /// Thread safety level the worker was created with.
    ucs_thread_mode_t threadMode{UCS_THREAD_MODE_MULTI};
    /// Maximum allowed header size for `ucp_am_send_nbx`.
    size_t maxAmHeader{0};
    /// Worker name used by tracing and analysis tools.
    std::string name{};
    /// Maximum debug-string buffer size accepted by `ucp_request_query`.
    size_t maxDebugString{0};
  };

  /**
   * @brief Get the worker's attributes.
   *
   * Returns the worker attributes as a struct, querying UCP via `ucp_worker_query` under
   * the hood. All non-address fields exposed by UCP are queried, use `addressBuilder()`
   * to obtain the address.
   *
   * @returns An `Attributes` filled with all queried fields.
   * @throws ucxx::Error if an error occurred while querying worker attributes.
   */
  [[nodiscard]] Attributes queryAttributes() const;
};

}  // namespace ucxx
