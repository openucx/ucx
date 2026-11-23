/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <utility>

#include <ucxx/typedefs.h>

namespace ucxx {

class Endpoint;
class MemoryHandle;
class RemoteKey;

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::RemoteKey>` objects.
 *
 * This class implements the builder pattern for `std::shared_ptr<ucxx::RemoteKey>`.
 * Construction happens when `build()` is called or when the builder is implicitly converted
 * to `std::shared_ptr<RemoteKey>`.
 */
class RemoteKeyBuilder final {
 public:
  /**
   * @brief Constructor for `RemoteKeyBuilder` from a local memory handle.
   *
   * @param[in] memoryHandle local memory handle to pack.
   */
  explicit RemoteKeyBuilder(std::shared_ptr<MemoryHandle> memoryHandle);

  /**
   * @brief Constructor for `RemoteKeyBuilder` from a serialized remote key.
   *
   * @param[in] endpoint endpoint used to unpack the remote key.
   * @param[in] serializedRemoteKey serialized remote key data.
   */
  RemoteKeyBuilder(std::shared_ptr<Endpoint> endpoint, SerializedRemoteKey serializedRemoteKey);

  /** @brief `RemoteKeyBuilder` destructor. */
  ~RemoteKeyBuilder();

  /** @brief Copy constructor (deep-copies internal state). */
  RemoteKeyBuilder(const RemoteKeyBuilder& other);
  /** @brief Copy assignment operator (deep-copies internal state). */
  RemoteKeyBuilder& operator=(const RemoteKeyBuilder& other);
  /** @brief Move constructor. */
  RemoteKeyBuilder(RemoteKeyBuilder&&) noexcept;
  /** @brief Move assignment operator. */
  RemoteKeyBuilder& operator=(RemoteKeyBuilder&&) noexcept;

  /**
   * @brief Implicit conversion operator to `shared_ptr<RemoteKey>`.
   *
   * @return The constructed `shared_ptr<ucxx::RemoteKey>` object.
   */
  operator std::shared_ptr<RemoteKey>();

  /**
   * @brief Build and return the `RemoteKey`.
   *
   * Each call to build() creates a new `RemoteKey` instance with the current parameters.
   *
   * @return The constructed `shared_ptr<ucxx::RemoteKey>` object.
   */
  [[nodiscard]] std::shared_ptr<RemoteKey> build();

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

}  // namespace ucxx
