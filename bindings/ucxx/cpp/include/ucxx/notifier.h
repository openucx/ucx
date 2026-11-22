/**
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <chrono>
#include <memory>

#include <ucp/api/ucp.h>

namespace ucxx {

/**
 * @brief The state of the notifier thread.
 *
 * The current state of the notifier thread.
 */
enum class RequestNotifierThreadState { NotRunning = 0, Running, Stopping };

/**
 * @brief The state with which a wait operation completed.
 *
 * The state with which a blocking call to wait for the request notifier completed.
 */
enum class RequestNotifierWaitState { Ready = 0, Timeout, Shutdown };

class Future;

/**
 * @brief Notifier for status of futures.
 *
 * A notifier used to delay notification of futures to a more appropriate stage of the
 * program execution, such as when it will be less resource intensive or free of risks of
 * effects such as deadlocks, for example when notifying Python futures where the GIL is
 * required.
 */
class Notifier {
 protected:
  Notifier() = default;

 public:
  Notifier(const Notifier&)            = delete;
  Notifier& operator=(Notifier const&) = delete;
  Notifier(Notifier&& o)               = delete;
  Notifier& operator=(Notifier&& o)    = delete;

  /**
   * @brief Virtual destructor.
   *
   * Virtual destructor with empty implementation.
   */
  virtual ~Notifier() {}

  /**
   * @brief Schedule notification of completed future.
   *
   * Schedule the notification of a completed Python future, but does not set the future's
   * result yet, which is later done by `runRequestNotifier()`. Because this call does
   * not notify the future, it does not require any resources associated with it.
   *
   * This is meant to be called from `ucxx::Future::notify()`.
   *
   * @param[in] future  future to notify.
   * @param[in] status  the request completion status.
   */
  virtual void scheduleFutureNotify(std::shared_ptr<Future> future, ucs_status_t status) = 0;

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
  virtual RequestNotifierWaitState waitRequestNotifier(uint64_t period) = 0;

  /**
   * @brief Notify event loop of all pending completed futures.
   *
   * This method will notify the internal resource of all pending completed futures.
   * Notifying the resource may require some exclusion mechanism, thus it should not run
   * indefinitely, but instead run periodically. Futures that completed must first be
   * scheduled with `scheduleFutureNotify()`.
   */
  virtual void runRequestNotifier() = 0;

  /**
   * @brief Make known to the notifier thread that it should stop.
   *
   * Often called when the application is shutting down, make known to the notifier that
   * it should stop and exit.
   */
  virtual void stopRequestNotifierThread() = 0;

  /**
   * @brief Returns whether the thread is running.
   *
   * @returns Whether the thread is running.
   */
  [[nodiscard]] virtual bool isRunning() const = 0;
};

}  // namespace ucxx
