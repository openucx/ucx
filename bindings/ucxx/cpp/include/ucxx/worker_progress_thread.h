/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include <ucxx/delayed_submission.h>

namespace ucxx {

/**
 * @brief A user-defined function used to wake the worker.
 *
 * A user-defined function signaling worker to wake the progress event.
 */
typedef std::function<void(void)> SignalWorkerFunction;

/**
 * @brief A user-defined function to execute at the start of the progress thread.
 *
 * A user-defined function to execute at the start of the progress thread, used for example
 * to ensure resources are properly set for the thread, such as a CUDA context.
 */
typedef std::function<void(void*)> ProgressThreadStartCallback;

/**
 * @brief Data for the user-defined function provided to progress thread start callback.
 *
 * Data passed to the user-defined function provided to the callback executed when the
 * progress thread starts, which the custom user-defined function may act upon.
 */
typedef void* ProgressThreadStartCallbackArg;

/**
 * @brief A thread to progress a `ucxx::Worker`.
 *
 * A thread to progress the `ucxx::Worker`, thus moving all such blocking operations out of
 * the main program thread. It may also be used to execute submissions, such as from
 * `ucxx::Request` objects, therefore also moving any blocking costs of those to this
 * thread, as well as generic pre-progress and post-progress callbacks that can be used by
 * the program to block until that stage is reached.
 */
class WorkerProgressThread {
 public:
  /**
   * @brief Progress thread stop synchronization configuration.
   */
  struct StopConfig {
    uint64_t callbackTimeoutNs;  ///< Maximum time to wait for each stop callback.
    uint64_t signalIntervalNs;   ///< Interval for waking the worker while waiting.

    /**
     * @brief Create stop synchronization configuration.
     *
     * @param[in] callbackTimeoutNs maximum time in nanoseconds to wait for each stop callback.
     * @param[in] signalIntervalNs interval in nanoseconds for waking the worker while waiting.
     *                             A value of 0 disables periodic wakeups: the worker is signaled
     *                             once before each wait and not again until the wait returns.
     */
    explicit StopConfig(uint64_t callbackTimeoutNs = 3000000000,
                        uint64_t signalIntervalNs  = 100000000)
      : callbackTimeoutNs(callbackTimeoutNs), signalIntervalNs(signalIntervalNs)
    {
    }
  };

 private:
  std::thread _thread{};  ///< Thread object
  std::shared_ptr<std::atomic<bool>> _stop{
    std::make_shared<std::atomic<bool>>(false)};  ///< Signal to stop on next iteration
  bool _pollingMode{false};  ///< Whether thread will use polling mode to progress
  SignalWorkerFunction _signalWorkerFunction{
    nullptr};  ///< Function signaling worker to wake the progress event (when _pollingMode is
               ///< `false`)
  ProgressThreadStartCallback _startCallback{
    nullptr};  ///< Callback to execute at start of the progress thread
  ProgressThreadStartCallbackArg _startCallbackArg{
    nullptr};  ///< Argument to pass to start callback
  std::shared_ptr<DelayedSubmissionCollection> _delayedSubmissionCollection{
    nullptr};  ///< Collection of enqueued delayed submissions

  /**
   * @brief The function executed in the new thread.
   *
   * This function ensures the `startCallback` is executed once at the start of the thread,
   * subsequently starting a continuous loop that processes any delayed submission requests
   * that are pending in the `delayedSubmissionCollection` followed by the execution of the
   * `progressFunction`, the loop repeats until `stop` is set.
   *
   * @param[in] progressFunction            user-defined progress function implementation.
   * @param[in] stop                        reference to the stop signal causing the
   *                                        progress loop to terminate.
   * @param[in] setThreadId                 callback function executed before the
   *                                        `startCallback` with a purpose of setting the
   *                                        thread ID with the parent so it is known before
   *                                        the progress loop starts.
   * @param[in] startCallback               user-defined callback function to be executed
   *                                        at the start of the progress thread.
   * @param[in] startCallbackArg            an argument to be passed to the start callback.
   * @param[in] delayedSubmissionCollection collection of delayed submissions to be
   *                                        processed during progress.
   */
  static void progressUntilSync(
    std::function<bool(void)> progressFunction,
    std::shared_ptr<std::atomic<bool>> stop,
    std::function<void(void)> setThreadId,
    ProgressThreadStartCallback startCallback,
    ProgressThreadStartCallbackArg startCallbackArg,
    std::shared_ptr<DelayedSubmissionCollection> delayedSubmissionCollection);

 public:
  WorkerProgressThread()                                       = default;
  WorkerProgressThread(const WorkerProgressThread&)            = delete;
  WorkerProgressThread& operator=(WorkerProgressThread const&) = delete;

  /**
   * @brief Move constructor of `WorkerProgressThread`.
   *
   * Transfers ownership of the progress thread resources from the source object to the
   * newly constructed object. The source object is left in a valid but unspecified state.
   *
   * @param[in] o the source object to move from.
   */
  WorkerProgressThread(WorkerProgressThread&& o) = default;

  /**
   * @brief Move assignment operator of `WorkerProgressThread`.
   *
   * Transfers ownership of the progress thread resources from the source object to the
   * target object. The source object is left in a valid but unspecified state.
   *
   * @param[in] o the source object to move from.
   * @returns reference to the target object.
   */
  WorkerProgressThread& operator=(WorkerProgressThread&& o) = default;

  /**
   * @brief Constructor of `WorkerProgressThread`.
   *
   * The constructor for a `WorkerProgressThread` object. The default constructor is
   * made private to ensure all UCXX objects are shared pointers for correct
   * lifetime management.
   *
   * This thread runs asynchronously with the main application thread. If you require
   * cross-thread synchronization (for example when tearing down the thread or canceling
   * requests), use the generic pre and post callbacks with a `CallbackNotifier` that
   * synchronizes with the application thread. Since the worker progress itself may change
   * state, it is usually the case that synchronization is needed in both pre and post
   * callbacks.
   *
   * @code{.cpp}
   * // context is `std::shared_ptr<ucxx::Context>`
   * auto worker = context->workerBuilder().build();
   *
   * // Equivalent to line above
   * // auto worker = ucxx::workerBuilder(context).build();
   * @endcode
   *
   * @param[in] pollingMode                 whether the thread should use polling mode to
   *                                        progress.
   * @param[in] progressFunction            user-defined progress function implementation.
   * @param[in] signalWorkerFunction        user-defined function to wake the worker
   *                                        progress event (when `pollingMode` is `false`).
   * @param[in] setThreadId                 callback function executed before the
   *                                        `startCallback` with a purpose of setting the
   *                                        thread ID with the parent so it is known before
   *                                        the progress loop starts.
   * @param[in] startCallback               user-defined callback function to be executed
   *                                        at the start of the progress thread.
   * @param[in] startCallbackArg            an argument to be passed to the start callback.
   * @param[in] delayedSubmissionCollection collection of delayed submissions to be
   *                                        processed during progress.
   */
  WorkerProgressThread(const bool pollingMode,
                       std::function<bool(void)> progressFunction,
                       std::function<void(void)> signalWorkerFunction,
                       std::function<void(void)> setThreadId,
                       ProgressThreadStartCallback startCallback,
                       ProgressThreadStartCallbackArg startCallbackArg,
                       std::shared_ptr<DelayedSubmissionCollection> delayedSubmissionCollection);

  /**
   * @brief `ucxx::WorkerProgressThread` destructor.
   *
   * Raises the stop signal and joins the thread.
   */
  ~WorkerProgressThread();

  /**
   * @brief Returns whether the thread was created for polling progress mode.
   *
   * @returns Whether polling mode is enabled.
   */
  [[nodiscard]] bool pollingMode() const;

  /**
   * @brief Returns the ID of the progress thread.
   *
   * @returns the progress thread ID.
   */
  [[nodiscard]] std::thread::id getId() const;

  /**
   * @brief Returns whether the thread is running.
   *
   * @returns Whether the thread is running.
   */
  [[nodiscard]] bool isRunning() const;

  /**
   * @brief Stop the progress thread.
   *
   * Raises the stop signal and joins the thread.
   *
   * @param[in] stopConfig  progress thread stop synchronization configuration.
   */
  void stop(StopConfig stopConfig = StopConfig{});
};

}  // namespace ucxx
