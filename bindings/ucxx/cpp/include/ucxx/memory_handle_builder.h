/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include <ucs/memory/memory_type.h>

namespace ucxx {

class Context;
class MemoryHandle;

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::MemoryHandle>` objects.
 *
 * This class implements the builder pattern for `std::shared_ptr<ucxx::MemoryHandle>`.
 * Construction happens when `build()` is called or when the builder is implicitly converted
 * to `std::shared_ptr<MemoryHandle>`.
 */
class MemoryHandleBuilder final {
 public:
  /**
   * @brief Constructor for `MemoryHandleBuilder`.
   *
   * @param[in] context parent context where memory is mapped.
   * @param[in] size minimum size of the memory allocation.
   */
  MemoryHandleBuilder(std::shared_ptr<Context> context, size_t size);

  /** @brief `MemoryHandleBuilder` destructor. */
  ~MemoryHandleBuilder();

  /** @brief Copy constructor (deep-copies internal state). */
  MemoryHandleBuilder(const MemoryHandleBuilder& other);
  /** @brief Copy assignment operator (deep-copies internal state). */
  MemoryHandleBuilder& operator=(const MemoryHandleBuilder& other);
  /** @brief Move constructor. */
  MemoryHandleBuilder(MemoryHandleBuilder&&) noexcept;
  /** @brief Move assignment operator. */
  MemoryHandleBuilder& operator=(MemoryHandleBuilder&&) noexcept;

  /**
   * @brief Implicit conversion operator to `shared_ptr<MemoryHandle>`.
   *
   * @return The constructed `shared_ptr<ucxx::MemoryHandle>` object.
   */
  operator std::shared_ptr<MemoryHandle>();

  /**
   * @brief Configure the buffer to map.
   *
   * @param[in] buffer pointer to an existing allocation or `nullptr` to allocate memory.
   * @return Reference to this builder for method chaining.
   */
  MemoryHandleBuilder& buffer(void* buffer);

  /**
   * @brief Configure the mapped memory type.
   *
   * @param[in] memoryType the type of memory the handle points to.
   * @return Reference to this builder for method chaining.
   */
  MemoryHandleBuilder& memoryType(ucs_memory_type_t memoryType);

  /**
   * @brief Build and return the `MemoryHandle`.
   *
   * Each call to build() creates a new `MemoryHandle` instance with the current parameters.
   *
   * @return The constructed `shared_ptr<ucxx::MemoryHandle>` object.
   */
  [[nodiscard]] std::shared_ptr<MemoryHandle> build();

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

}  // namespace ucxx
