/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/config.h>
#include <ucxx/context_builder.h>
#include <ucxx/memory_handle_builder.h>
#include <ucxx/worker_builder.h>

namespace ucxx {

class ContextBuilder;

class MemoryHandle;
class Worker;

/**
 * @brief Component encapsulating the UCP context.
 *
 * The UCP layer provides a handle to access its context in form of `ucp_context_h` object,
 * this class encapsulates that object and provides methods to simplify its handling.
 *
 * Use `ucxx::contextBuilder` to create and configure `Context` instances.
 */
class Context : public Component {
 private:
  ucp_context_h _handle{nullptr};  ///< The UCP context handle
  Config _config{{}};              ///< UCP context configuration variables
  uint64_t _featureFlags{0};       ///< Feature flags used to construct UCP context
  bool _cudaSupport{false};        ///< Whether CUDA support is enabled

  /**
   * @brief Private constructor of `shared_ptr<ucxx::Context>`.
   *
   * This is the internal implementation of `ucxx::Context` constructor, made private not
   * to be called directly.
   *
   * Instead the user should use `ucxx::contextBuilder()`.
   *
   * @param[in] ucxConfig configurations overriding `UCX_*` defaults and environment
   *                      variables.
   * @param[in] featureFlags feature flags to be used at UCP context construction time.
   */
  Context(const ConfigMap ucxConfig, const uint64_t featureFlags);

 public:
  static constexpr uint64_t defaultFeatureFlags =
    UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP | UCP_FEATURE_STREAM | UCP_FEATURE_AM |
    UCP_FEATURE_RMA;  ///< Suggested default context feature flags to use.

  Context()                          = delete;
  Context(const Context&)            = delete;
  Context& operator=(Context const&) = delete;
  Context(Context&& o)               = delete;
  Context& operator=(Context&& o)    = delete;

  // Allow internal construction without exposing factory functions.
  friend class detail::ConstructorFactory;

  /**
   * @brief Allow ContextBuilder to access private constructor.
   */
  friend class ContextBuilder;

  /**
   * @brief `ucxx::Context` destructor
   */
  ~Context();

  /**
   * @brief Get the context configuration.
   *
   * The context configuration is a `ConfigMap` containing entries of the UCX variables that were
   * set upon creation of the UCP context. Only those variables known to UCP can be acquired.
   *
   * @code{.cpp}
   *   // context is `std::shared_ptr<ucxx::Context>`
   *   auto contextConfig = context->getConfig();
   * @endcode
   *
   * @return A `ConfigMap` corresponding to the context's configuration.
   */
  [[nodiscard]] ConfigMap getConfig();

  /**
   * @brief Get the underlying `ucp_context_h` handle
   *
   * Lifetime of the `ucp_context_h` handle is managed by the `ucxx::Context`
   * object and its ownership is non-transferrable. Once the `ucxx::Context`
   * is destroyed the handle is not valid anymore, it is the user's
   * responsibility to ensure the owner's lifetime while using the handle.
   *
   * @code{.cpp}
   *   // context is `std::shared_ptr<ucxx::Context>`
   *   ucp_context_h contextHandle = context->getHandle();
   * @endcode
   *
   * @return The underlying `ucp_context_h` handle
   */
  [[nodiscard]] ucp_context_h getHandle();

  /**
   * @brief Get information from UCP context.
   *
   * Get information from UCP context, including memory domains, transport
   * resources, and other useful information. This method is a wrapper to
   * `ucp_context_print_info`.
   *
   * @code{.cpp}
   *   // context is `std::shared_ptr<ucxx::Context>`
   *   auto contextInfo = context->getInfo();
   * @endcode
   *
   * @return String containing context information
   */
  [[nodiscard]] std::string getInfo();

  /**
   * @brief Get feature flags that were used to construct the UCP context.
   *
   * Get feature flags that were used to construct the UCP context, this has
   * the same value that was specified by the user when creating the
   * `ucxx::Context` object.
   *
   * @code{.cpp}
   *   // context is `std::shared_ptr<ucxx::Context>`
   *   uint64_t contextFeatureFlags= context->getFeatureFlags();
   * @endcode
   *
   * @return Feature flags for this context
   */
  [[nodiscard]] uint64_t getFeatureFlags() const;

  /**
   * @brief Query whether CUDA support is available.
   *
   * Query whether the UCP context has CUDA support available. This is a done through a
   * combination of verifying whether CUDA memory support is available and `UCX_TLS` allows
   * CUDA to be enabled, essentially `UCX_TLS` must explicitly be one of the following:
   *
   * 1. Exactly `all`;
   * 2. Contain a field starting with `cuda`;
   * 3. Start with `^` (disable all listed transports) and _NOT_ contain a field named
   *    either `cuda` or `cuda_copy`.
   *
   * @return Whether CUDA support is available.
   */
  [[nodiscard]] bool hasCudaSupport() const;

  /**
   * @brief Create a builder for a new `ucxx::Worker`.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion constructs a `ucxx::Worker` as a child of this context.
   *
   * @returns Builder to configure optional worker parameters.
   */
  [[nodiscard]] WorkerBuilder workerBuilder();

  /**
   * @brief Create a builder for a new `ucxx::MemoryHandle`.
   *
   * Calling this method only creates the builder. Finalizing it with `.build()` or
   * implicit conversion maps memory as a child of this context.
   *
   * @param[in] size the minimum size of the memory allocation.
   * @returns Builder to configure optional memory mapping parameters.
   */
  [[nodiscard]] MemoryHandleBuilder memoryHandleBuilder(size_t size);
};

}  // namespace ucxx
