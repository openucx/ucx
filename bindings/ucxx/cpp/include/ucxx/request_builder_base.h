/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <stdexcept>
#include <utility>

#include <ucxx/typedefs.h>

namespace ucxx {

/**
 * @page ucxx_request_builder_pattern Request builder pattern
 *
 * Request builders capture the arguments required to create a request and expose optional
 * settings through fluent setters. Construction happens when `build()` is called or when the
 * builder is implicitly converted to a compatible `std::shared_ptr`.
 *
 * Request builders are move-only and single-use. Calling `build()` or implicitly converting
 * the same builder object more than once throws `std::logic_error`. The builder is consumed
 * before request validation or construction completes; if that first attempt throws, retrying
 * the same builder still throws `std::logic_error`.
 *
 * Request builder objects are not thread-safe. Accessing the same builder object from multiple
 * threads requires external synchronization. Distinct builder objects are independent.
 *
 * Each request-specific factory function returns the corresponding builder. Required arguments
 * are passed to the factory function, while optional settings such as Python future notification
 * and completion callbacks are configured through method chaining when supported by that request
 * type.
 */

/**
 * @brief CRTP base for all request builders providing Python future support.
 *
 * Holds `_enablePythonFuture` and provides the `pythonFuture()` fluent setter.
 * Derived builder classes inherit from this (or from `RequestCallbackBuilderBase`)
 * to avoid duplicating the member and setter in every builder.
 *
 * @tparam Derived  The concrete builder type (CRTP).
 */
template <typename Derived>
class RequestBuilderBase {
 protected:
  bool _enablePythonFuture{false};  ///< Enable Python future support
  bool _built{false};               ///< Whether build() has already been called

  /**
   * @brief Assert that build() has not been called yet, then mark as built.
   *
   * Builders are marked before request validation or construction completes, so a failed
   * first build attempt still consumes the builder.
   *
   * @throws std::logic_error if build() has already been called on this builder.
   */
  void markBuilt()
  {
    if (_built) throw std::logic_error("Builder::build() called more than once");
    _built = true;
  }

 public:
  /** @brief Default constructor. */
  RequestBuilderBase() = default;

  /** @brief Copy constructor disabled to prevent duplicate request submissions. */
  RequestBuilderBase(const RequestBuilderBase&) = delete;

  /** @brief Copy assignment disabled to prevent duplicate request submissions. */
  RequestBuilderBase& operator=(const RequestBuilderBase&) = delete;

  /** @brief Move constructor. */
  RequestBuilderBase(RequestBuilderBase&&) noexcept = default;

  /** @brief Move assignment operator. */
  RequestBuilderBase& operator=(RequestBuilderBase&&) = default;

  /**
   * @brief Configure Python future support.
   *
   * @param[in] enable whether a Python future should be created and notified (default: true).
   * @return Reference to this builder for method chaining.
   */
  Derived& pythonFuture(bool enable = true) &
  {
    _enablePythonFuture = enable;
    return static_cast<Derived&>(*this);
  }

  /**
   * @brief Configure Python future support on a temporary builder.
   *
   * @param[in] enable whether a Python future should be created and notified (default: true).
   * @return Rvalue reference to this builder for method chaining.
   */
  Derived&& pythonFuture(bool enable = true) &&
  {
    _enablePythonFuture = enable;
    return static_cast<Derived&&>(*this);
  }
};

/**
 * @brief CRTP base for request builders that support completion callbacks.
 *
 * Extends `RequestBuilderBase` with `_callbackFunction` / `_callbackData` members and
 * their fluent setters. Used by builders for `RequestTag`, `RequestAm`, `RequestMem`,
 * `RequestFlush`, and `RequestEndpointClose`.
 *
 * @tparam Derived  The concrete builder type (CRTP).
 */
template <typename Derived>
class RequestCallbackBuilderBase : public RequestBuilderBase<Derived> {
 protected:
  RequestCallbackUserFunction _callbackFunction{nullptr};  ///< User callback on completion
  RequestCallbackUserData _callbackData{nullptr};          ///< Data passed to callback

 public:
  /**
   * @brief Set the user-defined callback function to call upon completion.
   *
   * @param[in] fn user-defined callback function.
   * @return Reference to this builder for method chaining.
   */
  Derived& callbackFunction(RequestCallbackUserFunction fn) &
  {
    _callbackFunction = std::move(fn);
    return static_cast<Derived&>(*this);
  }

  /**
   * @brief Set the user-defined callback function on a temporary builder.
   *
   * @param[in] fn user-defined callback function.
   * @return Rvalue reference to this builder for method chaining.
   */
  Derived&& callbackFunction(RequestCallbackUserFunction fn) &&
  {
    _callbackFunction = std::move(fn);
    return static_cast<Derived&&>(*this);
  }

  /**
   * @brief Set the user-defined data to pass to the callback function.
   *
   * @param[in] data user-defined data passed to `callbackFunction`.
   * @return Reference to this builder for method chaining.
   */
  Derived& callbackData(RequestCallbackUserData data) &
  {
    _callbackData = std::move(data);
    return static_cast<Derived&>(*this);
  }

  /**
   * @brief Set the user-defined data to pass to the callback function on a temporary builder.
   *
   * @param[in] data user-defined data passed to `callbackFunction`.
   * @return Rvalue reference to this builder for method chaining.
   */
  Derived&& callbackData(RequestCallbackUserData data) &&
  {
    _callbackData = std::move(data);
    return static_cast<Derived&&>(*this);
  }
};

}  // namespace ucxx
