/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/component.h>
#include <ucxx/endpoint.h>
#include <ucxx/memory_handle.h>
#include <ucxx/remote_key_builder.h>

namespace ucxx {

class RemoteKey;

/**
 * @brief Type for hashing serialized remote keys.
 *
 * A size_t type used to store hash values of serialized remote keys, enabling
 * efficient comparison and lookup of remote key objects.
 */
typedef size_t SerializedRemoteKeyHash;

/**
 * @brief Component holding a UCP rkey (remote key).
 *
 * To provide RMA (Remote Memory Access) to memory handles, UCP packs their information
 * in the form of `ucp_rkey_h` (remote key, or rkey for short). This class encapsulates
 * that object and provides methods to simplify its handling, both locally and remotely
 * including (de-)serialization for transfers over the wire and reconstruction of the
 * object on the remote process.
 */
class RemoteKey : public Component {
 private:
  ucp_rkey_h _remoteKey{nullptr};  ///< The unpacked remote key.
  void* _packedRemoteKey{
    nullptr};  ///< The packed ucp_rkey_h key, suitable for transfer to a remote process.
  size_t _packedRemoteKeySize{0};              ///< The size in bytes of the remote key.
  std::vector<char> _packedRemoteKeyVector{};  ///< The deserialized packed remote key.
  uint64_t _memoryBaseAddress{0};              ///< The allocation's base address.
  size_t _memorySize{0};                       ///< The actual allocation size.

  /**
   * @brief Private constructor of `ucxx::RemoteKey`.
   *
   * This is the internal implementation of `ucxx::RemoteKey` constructor from a local
   * `std::shared_ptr<ucxx::MemoryHandle>`, made private not to be called directly. This
   * constructor is made private to ensure all UCXX objects are shared pointers and the
   * correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::MemoryHandle::remoteKeyBuilder()`
   * - `ucxx::RemoteKeyBuilder`
   *
   * @param[in] memoryHandle the memory handle mapped on the local process.
   */
  explicit RemoteKey(std::shared_ptr<MemoryHandle> memoryHandle);

  /**
   * @brief Private constructor of `ucxx::RemoteKey`.
   *
   * This is the internal implementation of `ucxx::RemoteKey` constructor from a remote
   * `std::shared_ptr<ucxx::MemoryHandle>`, made private not to be called directly. This
   * constructor is made private to ensure all UCXX objects are shared pointers and the
   * correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::Endpoint::remoteKeyBuilder()`
   * - `ucxx::RemoteKeyBuilder`
   *
   * @param[in] endpoint            the `std::shared_ptr<Endpoint>` parent component.
   * @param[in] serializedRemoteKey the remote key that was serialized by the owner of
   *                                the memory handle and transferred over-the-wire for
   *                                reconstruction and remote access.
   */
  RemoteKey(std::shared_ptr<Endpoint> endpoint, SerializedRemoteKey serializedRemoteKey);

  /**
   * @brief Deserialize and reconstruct the remote key.
   *
   * Deserialize the remote key that was serialized with `ucxx::RemoteKey::serialize()` and
   * possibly transferred over-the-wire and reconstruct the object to allow remote access.
   *
   * @code{.cpp}
   * // remoteKey is `std::shared_ptr<ucxx::RemoteKey>`
   * auto serializedRemoteKey = remoteKey->serialize();
   * @endcode
   *
   * @throws std::runtime_error if checksum of the serialized object fails.
   *
   * @returns The deserialized remote key.
   */
  void deserialize(const SerializedRemoteKey& serializedHeader);

 public:
  // Allow internal construction without exposing factory functions.
  friend class detail::ConstructorFactory;

  ~RemoteKey();

  /**
   * @brief Get the underlying `ucp_rkey_h` handle.
   *
   * Lifetime of the `ucp_rkey_h` handle is managed by the `ucxx::RemoteKey` object and
   * its ownership is non-transferrable. Once the `ucxx::RemoteKey` is destroyed the handle
   * becomes invalid and so does the address to the remote memory handle it points to, it is
   * the user's responsibility to ensure the owner's lifetime while using the handle.
   *
   * @code{.cpp}
   * // remoteKey is `std::shared_ptr<ucxx::RemoteKey>`
   * auto remoteKeyHandle = remoteKey->getHandle();
   * @endcode
   *
   * @returns The underlying `ucp_mem_h` handle.
   */
  [[nodiscard]] ucp_rkey_h getHandle();

  /**
   * @brief Get the size of the memory allocation.
   *
   * Get the size of the memory allocation the remote key packs, which is at least the
   * number of bytes specified with the `size` argument passed to `memoryHandleBuilder()`.
   *
   * @code{.cpp}
   * // remoteKey is `std::shared_ptr<ucxx::RemoteKey>`
   * auto remoteMemorySize = remoteKey->getSize();
   * @endcode
   *
   * @returns The size of the memory allocation.
   */
  [[nodiscard]] size_t getSize() const;

  /**
   * @brief Get the base address of the memory allocation.
   *
   * Get the base address of the memory allocation the remote key packs, which is going
   * to be used as the remote address to put or get memory via the
   * `ucxx::Endpoint::memPutBuilder()` or `ucxx::Endpoint::memGetBuilder()` methods.
   *
   * @code{.cpp}
   * // remoteKey is `std::shared_ptr<ucxx::RemoteKey>`
   * auto remoteMemoryBaseAddress = remoteKey->getBaseAddress();
   * @endcode
   *
   * @returns The base address of the memory allocation.
   */
  [[nodiscard]] uint64_t getBaseAddress();

  /**
   * @brief Serialize the remote key.
   *
   * Serialize the remote key to allow over-the-wire transfer and subsequent
   * reconstruction of the object in the remote process.
   *
   * @code{.cpp}
   * // remoteKey is `std::shared_ptr<ucxx::RemoteKey>`
   * auto serializedRemoteKey = remoteKey->serialize();
   * @endcode
   *
   * @returns The serialized remote key.
   */
  [[nodiscard]] SerializedRemoteKey serialize() const;
};

}  // namespace ucxx
