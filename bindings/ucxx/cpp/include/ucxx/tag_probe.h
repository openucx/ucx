/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <atomic>
#include <memory>
#include <optional>

#include <ucp/api/ucp.h>
#include <ucxx/tag_probe_builder.h>
#include <ucxx/typedefs.h>

namespace ucxx {

namespace detail {
class ConstructorFactory;
}

// Forward declaration for friend access
class RequestTag;
class TagProbeInfo;

/**
 * @brief Information about probed tag message.
 *
 * Contains information returned when probing by a tag message received by the worker but
 * not yet consumed.
 */
class TagRecvInfo {
 public:
  const Tag senderTag;  ///< Sender tag
  const size_t length;  ///< The size of the received data

  /**
   * @brief Default constructor for TagRecvInfo.
   */
  TagRecvInfo() : senderTag(Tag(0)), length(0) {}

  /**
   * @brief Construct a TagRecvInfo object from a UCP tag receive info structure.
   *
   * @param[in] info  The UCP tag receive info structure containing the sender tag and
   *                  received data length.
   */
  explicit TagRecvInfo(const ucp_tag_recv_info_t& info);
};

/**
 * @brief Information about probed tag message.
 *
 * Contains complete information about a probed tag message, including whether a message
 * was matched, the tag receive information, and the message handle for efficient reception
 * when `tagProbe()` is called with `remove=true`.
 *
 * @warning Callers must check `isMatched()` before calling `getInfo()` or `getHandle()`
 *          to prevent misuse and undefined behavior. When `isMatched()` returns `false`,
 *          `getInfo()` and `getHandle()` will throw std::runtime_error.
 *
 * @note This class is managed via `shared_ptr` without support for copy or move semantics
 * to prevent multiple objects from holding the same handle, which could lead to undefined
 * behavior if the message is received multiple times.
 */
class TagProbeInfo {
 public:
  TagProbeInfo(const TagProbeInfo&)            = delete;
  TagProbeInfo& operator=(TagProbeInfo const&) = delete;
  TagProbeInfo(TagProbeInfo&& o)               = delete;
  TagProbeInfo& operator=(TagProbeInfo&& o)    = delete;

  ~TagProbeInfo();

  /**
   * @brief Check if a message was matched.
   *
   * @returns true if a message was matched, false otherwise.
   */
  bool isMatched() const;

  /**
   * @brief Get tag receive information.
   *
   * @throws std::runtime_error if no message was matched.
   * @returns The tag receive information.
   */
  const TagRecvInfo& getInfo() const;

  /**
   * @brief Get the message handle.
   *
   * @throws std::runtime_error if the handle is nullptr or has been consumed.
   * @returns The UCP tag message handle.
   */
  ucp_tag_message_h getHandle() const;

  /**
   * @brief Release the message handle.
   *
   * Same as `getHandle()` but also marks the handle as consumed, therefore no
   * warning is logged when the object is destroyed and is the user's responsibility
   * to consume the handle via a direct UCP operation.
   *
   * @throws std::runtime_error if the handle is nullptr or has been consumed.
   * @returns The UCP tag message handle.
   */
  ucp_tag_message_h releaseHandle() const;

 private:
  const bool _matched{false};  ///< Whether a message was matched
  const std::optional<TagRecvInfo> _info{
    std::nullopt};  ///< Tag receive information (only valid if `isMatched()` returns `true`)
  const ucp_tag_message_h _handle{
    nullptr};  ///< Message handle (only valid if `isMatched()` returns `true` and
               ///< `tagProbe()` is called with `remove=true`)
  mutable std::atomic<bool> _consumed{false};  ///< Whether the message has been consumed

  /**
   * @brief Mark the handle as consumed.
   *
   * This method is private and can only be called by RequestTag via friend access.
   * It's called automatically when the handle is used for a UCP operation.
   */
  void consume() const;

  /**
   * @brief Private constructor of `ucxx::TagProbeInfo`.
   *
   * Initializes the object as unmatched.
   *
   * This is the internal implementation of `ucxx::TagProbeInfo` default constructor, made
   * private not to be called directly. This constructor is made private to ensure all UCXX
   * objects are shared pointers and the correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::TagProbeInfoBuilder`
   */
  TagProbeInfo() = default;

  /**
   * @brief Private constructor of `ucxx::TagProbeInfo`.
   *
   * Initializes the object as matched with the provided info and handle.
   *
   * This is the internal implementation of `ucxx::TagProbeInfo` default constructor, made
   * private not to be called directly. This constructor is made private to ensure all UCXX
   * objects are shared pointers and the correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::TagProbeInfoBuilder`
   *
   * @param[in] info    The UCP tag receive info structure.
   * @param[in] handle  The UCP tag message handle (can be nullptr if `tagProbe()` is called
   *                    with `remove=false`).
   */
  TagProbeInfo(const ucp_tag_recv_info_t& info, ucp_tag_message_h handle);

  // Allow internal construction without exposing factory functions.
  friend class detail::ConstructorFactory;

  // Allow RequestTag to consume the handle
  friend class RequestTag;
};

}  // namespace ucxx
