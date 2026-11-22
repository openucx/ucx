/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>

#include <ucxx/typedefs.h>

namespace ucxx {

// Forward declarations
class Context;

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::Context>` objects.
 *
 * This class implements the builder pattern for `std::shared_ptr<ucxx::Context>`, allowing
 * optional parameters to be specified via method chaining. Construction happens immediately
 * when the builder expression completes, ensuring one-time construction with immediate
 * evaluation.
 *
 * The feature flags are required and must be provided to `contextBuilder()`.
 * The `configMap()` method is optional.
 *
 * `ContextBuilder` objects are not thread-safe. Accessing the same builder object from multiple
 * threads requires external synchronization. Distinct builder objects are independent.
 *
 * @code{.cpp}
 *   // Minimal usage (only featureFlags required)
 *   auto ctx = ucxx::contextBuilder(UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP).build();
 *
 *   // With optional configMap
 *   auto context = ucxx::contextBuilder(UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP)
 *                    .configMap({{"TLS", "tcp"}})
 *                    .build();
 *
 *   // Using implicit conversion
 *   std::shared_ptr<ucxx::Context> ctx = ucxx::contextBuilder(UCP_FEATURE_RMA);
 * @endcode
 */
class ContextBuilder final {
 public:
  /**
   * @brief Constructor for `ContextBuilder` with required feature flags.
   *
   * @param[in] featureFlags feature flags to be used at UCP context construction time (required).
   */
  explicit ContextBuilder(uint64_t featureFlags);

  /**
   * @brief `ContextBuilder` destructor.
   */
  ~ContextBuilder();

  /** @brief Copy constructor (deep-copies internal state). */
  ContextBuilder(const ContextBuilder& other);
  /** @brief Copy assignment operator (deep-copies internal state). */
  ContextBuilder& operator=(const ContextBuilder& other);
  /** @brief Move constructor. */
  ContextBuilder(ContextBuilder&&) noexcept;
  /** @brief Move assignment operator. */
  ContextBuilder& operator=(ContextBuilder&&) noexcept;

  /**
   * @brief Implicit conversion operator to `shared_ptr<Context>`.
   *
   * Enables automatic construction of the `Context` when the builder is used in a context
   * requiring a `shared_ptr<Context>`, delegating to `build()`.
   *
   * @return The constructed `shared_ptr<ucxx::Context>` object.
   */
  operator std::shared_ptr<Context>();

  /**
   * @brief Set the configuration map for the context.
   *
   * @param[in] configMap configurations overriding `UCX_*` defaults and environment variables.
   * @return Reference to this builder for method chaining.
   */
  ContextBuilder& configMap(ConfigMap configMap);

  /**
   * @brief Build and return the `Context`.
   *
   * This method constructs the `Context` with the specified parameters and returns it.
   * Each call to build() creates a new `Context` instance with the current parameters.
   *
   * @return The constructed `shared_ptr<ucxx::Context>` object.
   */
  [[nodiscard]] std::shared_ptr<Context> build();

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

/**
 * @brief Create a ContextBuilder for constructing a `shared_ptr<ucxx::Context>`.
 *
 * This function returns a builder object that allows setting optional context parameters
 * via method chaining. The feature flags are required and must be provided as an argument.
 * The actual context is constructed when the builder is converted to a `shared_ptr<Context>`
 * or when `build()` is called.
 *
 * @code{.cpp}
 *   // Minimal usage (only featureFlags required)
 *   auto context = ucxx::contextBuilder(UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP).build();
 *
 *   // With optional configMap
 *   auto context = ucxx::contextBuilder(UCP_FEATURE_TAG)
 *                    .configMap({{"TLS", "tcp"}})
 *                    .build();
 *
 *   // Using implicit conversion
 *   std::shared_ptr<ucxx::Context> context = ucxx::contextBuilder(UCP_FEATURE_RMA);
 * @endcode
 *
 * @param[in] featureFlags feature flags to be used at UCP context construction time (required).
 * @return A ContextBuilder object that can be used to set optional parameters.
 */
[[nodiscard]] inline ContextBuilder contextBuilder(uint64_t featureFlags)
{
  return ContextBuilder(featureFlags);
}

}  // namespace ucxx
