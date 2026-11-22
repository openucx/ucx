/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <string_view>

#include <ucp/api/ucp.h>

#include <ucxx/address_builder.h>
#include <ucxx/component.h>
#include <ucxx/worker.h>

namespace ucxx {

class Address;

/**
 * @brief Component encapsulating the address of a UCP worker.
 *
 * A UCP worker has a unique address that can is contained in a `ucp_address_t*` object,
 * this class encapsulates that object and provides methods to simplify its handling.
 */
class Address : public Component {
 private:
  ucp_address_t* _handle{nullptr};
  size_t _length{0};

  /**
   * @brief Private constructor of `ucxx::Address`.
   *
   * This is the internal implementation of `ucxx::Address` constructor, made private not
   * to be called directly. This constructor is made private to ensure all UCXX objects
   * are shared pointers and the correct lifetime management of each one.
   *
   * Instead the user should use one of the following:
   *
   * - `ucxx::AddressBuilder`
   * - `ucxx::Worker::addressBuilder()`
   *
   * @param[in] worker  the parent `set::shared_ptr<Worker>` component.
   * @param[in] address UCP address handle.
   * @param[in] length  length of the address byte-string in bytes.
   */
  Address(std::shared_ptr<Worker> worker, ucp_address_t* address, size_t length);

 public:
  Address()                          = delete;
  Address(const Address&)            = delete;
  Address& operator=(Address const&) = delete;
  Address(Address&& o)               = delete;
  Address& operator=(Address&& o)    = delete;

  ~Address();

  // Allow internal construction without exposing factory functions.
  friend class detail::ConstructorFactory;

  /**
   * @brief Get the underlying `ucp_address_t*` handle.
   *
   * Lifetime of the `ucp_address_t*` handle is managed by the `ucxx::Address` object and
   * its ownership is non-transferrable. Once the `ucxx::Address` is destroyed the handle
   * is not valid anymore, it is the user's responsibility to ensure the owner's lifetime
   * while using the handle.
   *
   * @code{.cpp}
   * // address is `std::shared_ptr<ucxx::Address>`
   * ucp_address_t* addressHandle = address->getHandle();
   * @endcode
   *
   * @returns The underlying `ucp_address_t` handle.
   */
  [[nodiscard]] ucp_address_t* getHandle() const;

  /**
   * @brief Get the length of the `ucp_address_t*` handle.
   *
   * Get the length of the `ucp_address_t*` handle, required to access the complete address
   * and prevent reading out-of-bound.
   *
   * @returns The length of the `ucp_address_t*` handle in bytes.
   */
  [[nodiscard]] size_t getLength() const;

  /**
   * @brief Get a non-owning view of the address as a string.
   *
   * Convenience method to view the underlying address as a `std::string_view` and return it as
   * a single object.
   *
   * @note If the user would like to own the object, they can pass the result to `std::string`,
   * which will copy the memory.
   *
   * @returns The underlying `ucp_address_t` handle.
   */
  [[nodiscard]] std::string_view getStringView() const;
};

}  // namespace ucxx
