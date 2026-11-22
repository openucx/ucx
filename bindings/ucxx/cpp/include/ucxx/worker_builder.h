/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <utility>

#include <ucxx/buffer.h>

namespace ucxx {

// Forward declarations
class Context;
class Worker;

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::Worker>` objects.
 *
 * This class implements the builder pattern for `std::shared_ptr<ucxx::Worker>`, allowing
 * optional parameters to be specified via method chaining. Construction happens when the
 * builder expression completes (via implicit conversion) or when `build()` is called.
 *
 * The `context` is required and must be provided to `workerBuilder()`.
 * The `delayedSubmission()` and `pythonFuture()` methods are optional.
 *
 * `WorkerBuilder` objects are not thread-safe. Accessing the same builder object from multiple
 * threads requires external synchronization. Distinct builder objects are independent.
 *
 * @code{.cpp}
 *   // Minimal usage (no options enabled)
 *   auto worker = ucxx::workerBuilder(context).build();
 *
 *   // With optional parameters
 *   auto worker = ucxx::workerBuilder(context)
 *                   .delayedSubmission(true)
 *                   .pythonFuture(true)
 *                   .build();
 *
 *   // Using implicit conversion
 *   std::shared_ptr<ucxx::Worker> worker = ucxx::workerBuilder(context);
 * @endcode
 */
class WorkerBuilder final {
 public:
  /**
   * @brief Constructor for `WorkerBuilder` with required context.
   *
   * @param[in] context context from which the worker will be created (required).
   */
  explicit WorkerBuilder(std::shared_ptr<Context> context);

  /**
   * @brief `WorkerBuilder` destructor.
   */
  ~WorkerBuilder();

  /** @brief Copy constructor (deep-copies internal state). */
  WorkerBuilder(const WorkerBuilder& other);
  /** @brief Copy assignment operator (deep-copies internal state). */
  WorkerBuilder& operator=(const WorkerBuilder& other);
  /** @brief Move constructor. */
  WorkerBuilder(WorkerBuilder&&) noexcept;
  /** @brief Move assignment operator. */
  WorkerBuilder& operator=(WorkerBuilder&&) noexcept;

  /**
   * @brief Implicit conversion operator to `shared_ptr<Worker>`.
   *
   * Enables automatic construction of the `Worker` when the builder is used in a context
   * requiring a `shared_ptr<Worker>`, delegating to `build()`.
   *
   * @return The constructed `shared_ptr<ucxx::Worker>` object.
   */
  operator std::shared_ptr<Worker>();

  /**
   * @brief Configure delayed submission to the progress thread.
   *
   * @param[in] enable whether delayed submission is enabled (default: true).
   * @return Reference to this builder for method chaining.
   */
  WorkerBuilder& delayedSubmission(bool enable = true);

  /**
   * @brief Configure Python future support.
   *
   * @param[in] enable whether Python futures are enabled (default: true).
   * @return Reference to this builder for method chaining.
   */
  WorkerBuilder& pythonFuture(bool enable = true);

  /**
   * @brief Configure request attributes querying.
   *
   * When enabled, each `ucxx::Request` created from the worker will have its UCP
   * attributes (such as the debug string) queried immediately after submission, making
   * them available via `ucxx::Request::getRequestAttributes()`. This may have
   * non-negligible runtime cost and is therefore disabled by default.
   *
   * @param[in] enable whether request attributes querying is enabled (default: true).
   * @return Reference to this builder for method chaining.
   */
  WorkerBuilder& requestAttributes(bool enable = true);

  /**
   * @brief Configure the preferred buffer type for CUDA allocations.
   *
   * @param[in] bufferType the preferred buffer type for CUDA allocations.
   * @return Reference to this builder for method chaining.
   */
  WorkerBuilder& cudaBufferType(BufferType bufferType);

  /**
   * @brief Build and return the `Worker`.
   *
   * This method constructs the `Worker` with the specified parameters and returns it.
   * Each call to build() creates a new `Worker` instance with the current parameters.
   *
   * @return The constructed `shared_ptr<ucxx::Worker>` object.
   */
  [[nodiscard]] std::shared_ptr<Worker> build();

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

/**
 * @brief Create a WorkerBuilder for constructing a `shared_ptr<ucxx::Worker>`.
 *
 * This function returns a builder object that allows setting optional worker parameters
 * via method chaining. The context is required and must be provided as an argument. The
 * actual worker is constructed when the builder is converted to a `shared_ptr<Worker>`
 * or when `build()` is called.
 *
 * @code{.cpp}
 *   auto worker = ucxx::workerBuilder(context)
 *                   .delayedSubmission()
 *                   .build();
 * // Minimal usage (only context required)
 * auto worker = ucxx::workerBuilder(context).build();
 *
 * // With optional enableDelayedSubmission
 * auto worker = ucxx::workerBuilder(context)
 *                 .delayedSubmission()
 *                 .build();
 *
 * // Using implicit conversion
 * std::shared_ptr<ucxx::Worker> worker = ucxx::workerBuilder(context);
 *
 * @endcode
 *
 * @param[in] context context from which the worker will be created (required).
 * @return A WorkerBuilder object that can be used to set optional parameters.
 */
[[nodiscard]] inline WorkerBuilder workerBuilder(std::shared_ptr<Context> context)
{
  return WorkerBuilder(std::move(context));
}

}  // namespace ucxx
