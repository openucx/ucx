/**
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>

#include <ucp/api/ucp.h>

#include <ucxx/notifier.h>

namespace ucxx {

/**
 * @brief Represent a future that may be notified by a specialized notifier.
 *
 * Represent a future object that may postpone notification of its status to a more
 * appropriate stage by a specialize notifier, such as `ucxx::Notifier`.
 */
class Future : public std::enable_shared_from_this<Future> {
 protected:
  std::shared_ptr<Notifier> _notifier{nullptr};  ///< The notifier object

  /**
   * @brief Construct a future that may be notified from a notifier object.
   *
   * Construct a future that may be notified from a notifier object, usually running
   * on a separate thread to decrease overhead from the application thread.
   *
   * This class may also be used to set the result or exception from any thread.
   *
   * @param[in] notifier  notifier object, possibly running on a separate thread.
   */
  explicit Future(std::shared_ptr<Notifier> notifier) : _notifier(notifier) {}

 public:
  Future()                         = delete;
  Future(const Future&)            = delete;
  Future& operator=(Future const&) = delete;
  Future(Future&& o)               = delete;
  Future& operator=(Future&& o)    = delete;

  /**
   * @brief Virtual destructor.
   *
   * Virtual destructor with empty implementation.
   */
  virtual ~Future() {}

  /**
   * @brief Inform the notifier that the future has completed.
   *
   * Inform the notifier that the future has completed so it can notify associated
   * resources of that occurrence.
   *
   * @throws std::runtime_error if the object is invalid or has been already released.
   *
   * @param[in] status  request completion status.
   */
  virtual void notify(ucs_status_t status) = 0;

  /**
   * @brief Set the future completion status.
   *
   * Set the future status as completed, either with a successful completion or error.
   *
   * @throws std::runtime_error if the object is invalid or has been already released.
   *
   * @param[in] status  request completion status.
   */
  virtual void set(ucs_status_t status) = 0;

  /**
   * @brief Get the underlying handle but does not release ownership.
   *
   * Get the underlying handle without releasing ownership. This can be useful for example
   * for logging, where we want to see the address of the pointer but do not want to
   * transfer ownership.
   *
   * @warning The destructor will also destroy the future, a pointer taken via this method
   * will cause the object to become invalid.
   *
   * @throws std::runtime_error if the object is invalid or has been already released.
   *
   * @returns The underlying handle.
   */
  [[nodiscard]] virtual void* getHandle() = 0;

  /**
   * @brief Get the underlying handle and release ownership.
   *
   * Get the underlying handle releasing ownership. This should be used when the future
   * needs to be permanently transferred to consumer code. After calling this method the
   * object becomes invalid for any other uses.
   *
   * @throws std::runtime_error if the object is invalid or has been already released.
   *
   * @returns The underlying handle.
   */
  [[nodiscard]] virtual void* release() = 0;
};

}  // namespace ucxx
