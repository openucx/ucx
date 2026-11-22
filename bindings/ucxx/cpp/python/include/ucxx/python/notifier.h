/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <ucxx/future.h>
#include <ucxx/notifier.h>

namespace ucxx {

namespace python {

/**
 * @brief Specialized Python implementation of a `ucxx::Notifier`.
 *
 * Specialized Python implementation of a `ucxx::Notifier`, providing support for notifying
 * Python futures.
 */
class Notifier : public ::ucxx::Notifier {
 private:
  std::mutex _notifierThreadMutex{};  ///< Mutex to access thread's resources
  std::vector<std::pair<std::shared_ptr<::ucxx::Future>, ucs_status_t>>
    _notifierThreadFutureStatus{};               ///< Container with futures and statuses to set
  bool _notifierThreadFutureStatusReady{false};  ///< Whether a future is scheduled for notification
  RequestNotifierThreadState _notifierThreadFutureStatusFinished{
    RequestNotifierThreadState::NotRunning};  ///< State of the notifier thread
  std::condition_variable
    _notifierThreadConditionVariable{};  ///< Condition variable used to wait for event

  /**
   * @brief Private constructor of `ucxx::python::Notifier`.
   *
   * This is the internal `ucxx::python::Notifier` constructor, made private not to be
   * called directly. Instead the user should call `ucxx::python::createNotifier()`.
   */
  Notifier() = default;

  /**
   * @brief Wait for a new event without a timeout.
   *
   * Block while waiting for an event (new future to be notified or stop signal)
   * indefinitely.
   *
   * @warning Use with caution, if no event ever occurs it will be impossible to continue
   * the thread.
   */
  [[nodiscard]] RequestNotifierWaitState waitRequestNotifierWithoutTimeout();

  /**
   * @brief Wait for a new event with a timeout.
   *
   * Block while waiting for an event (new future to be notified or stop signal) with added
   * timeout to unblock after a certain period if no event has occurred.
   *
   * @param[in] period the time to wait for an event before unblocking.
   */
  [[nodiscard]] RequestNotifierWaitState waitRequestNotifierWithTimeout(uint64_t period);

 public:
  Notifier(const Notifier&)            = delete;
  Notifier& operator=(Notifier const&) = delete;
  Notifier(Notifier&& o)               = delete;
  Notifier& operator=(Notifier&& o)    = delete;

  /**
   * @brief Constructor of `shared_ptr<ucxx::python::Notifier>`.
   *
   * The constructor for a `shared_ptr<ucxx::python::Notifier>` object. The default
   * constructor is made private to ensure all UCXX objects are shared pointers for correct
   * lifetime management.
   *
   * The notifier should run on its own Python thread, but need to have the same asyncio
   * event loop set as the application thread. By running a notifier on its own thread the
   * application thread can be decoupled from the overhead of allowing the UCX worker to
   * progress on the same thread as the application to be able to notify each future, as
   * removing the requirement for the GIL at any time by the UCX backend.
   *
   * @returns The `shared_ptr<ucxx::python::Notifier>` object
   */
  friend std::shared_ptr<::ucxx::Notifier> createNotifier();

  /**
   * @brief Virtual destructor.
   *
   * Virtual destructor with empty implementation.
   */
  virtual ~Notifier();

  /**
   * @brief Schedule event loop notification of completed Python future.
   *
   * Schedule the notification of the event loop of a completed Python future, but does
   * not notify the event loop yet, which is later done by `runRequestNotifier()`. Because
   * this call does not notify the Python asyncio event loop, it does not require the GIL
   * to execute.
   *
   * This is meant to be called from `ucxx::python::Future::notify()`.
   *
   * @param[in] future  Python future to notify.
   * @param[in] status  the request completion status.
   */
  void scheduleFutureNotify(std::shared_ptr<::ucxx::Future> future, ucs_status_t status) override;

  /**
   * @brief Wait for a new event with a timeout in nanoseconds.
   *
   * Block while waiting for an event (new future to be notified or stop signal) with added
   * timeout in nanoseconds to unblock after a that period if no event has occurred. A
   * period of zero means this call will never unblock until an event occurs.
   *
   * @warning Be cautious using a period of zero, if no event ever occurs it will be
   * impossible to continue the thread.
   *
   * @param[in] period the time in nanoseconds to wait for an event before unblocking.
   */
  [[nodiscard]] RequestNotifierWaitState waitRequestNotifier(uint64_t period) override;

  /**
   * @brief Notify event loop of all pending completed Python futures.
   *
   * This method will notify the Python asyncio event loop of all pending completed
   * futures. Notifying the event loop requires taking the Python GIL, thus it cannot run
   * indefinitely but must instead run periodically. Futures that completed must first be
   * scheduled with `scheduleFutureNotify()`.
   */
  void runRequestNotifier() override;

  /**
   * @brief Make known to the notifier thread that it should stop.
   *
   * Often called when the application is shutting down, make known to the notifier thread
   * that it should stop and exit.
   */
  void stopRequestNotifierThread() override;

  /**
   * @brief Returns whether the thread is running.
   *
   * @returns Whether the thread is running.
   */
  [[nodiscard]] bool isRunning() const override;
};

}  // namespace python

}  // namespace ucxx
