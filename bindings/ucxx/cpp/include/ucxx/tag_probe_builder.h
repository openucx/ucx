/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>

#include <ucp/api/ucp.h>

namespace ucxx {

class TagProbeInfo;

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::TagProbeInfo>` objects.
 *
 * This class implements the builder pattern for `std::shared_ptr<ucxx::TagProbeInfo>`.
 * Construction happens when `build()` is called or when the builder is implicitly converted
 * to `std::shared_ptr<TagProbeInfo>`.
 */
class TagProbeInfoBuilder final {
 public:
  /**
   * @brief Constructor for an unmatched `TagProbeInfoBuilder`.
   */
  TagProbeInfoBuilder();

  /**
   * @brief Constructor for a matched `TagProbeInfoBuilder`.
   *
   * @param[in] info UCP tag receive info.
   * @param[in] handle UCP tag message handle.
   */
  TagProbeInfoBuilder(const ucp_tag_recv_info_t& info, ucp_tag_message_h handle);

  /** @brief `TagProbeInfoBuilder` destructor. */
  ~TagProbeInfoBuilder();

  /** @brief Copy constructor (deep-copies internal state). */
  TagProbeInfoBuilder(const TagProbeInfoBuilder& other);
  /** @brief Copy assignment operator (deep-copies internal state). */
  TagProbeInfoBuilder& operator=(const TagProbeInfoBuilder& other);
  /** @brief Move constructor. */
  TagProbeInfoBuilder(TagProbeInfoBuilder&&) noexcept;
  /** @brief Move assignment operator. */
  TagProbeInfoBuilder& operator=(TagProbeInfoBuilder&&) noexcept;

  /**
   * @brief Implicit conversion operator to `shared_ptr<TagProbeInfo>`.
   *
   * @return The constructed `shared_ptr<ucxx::TagProbeInfo>` object.
   */
  operator std::shared_ptr<TagProbeInfo>();

  /**
   * @brief Build and return the `TagProbeInfo`.
   *
   * Each call to build() creates a new `TagProbeInfo` instance with the current parameters.
   *
   * @return The constructed `shared_ptr<ucxx::TagProbeInfo>` object.
   */
  [[nodiscard]] std::shared_ptr<TagProbeInfo> build();

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

}  // namespace ucxx
