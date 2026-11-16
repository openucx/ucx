/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <string_view>
#include <utility>

namespace ucxx {

class Address;
class Worker;

/**
 * @brief Builder class for constructing `std::shared_ptr<ucxx::Address>` objects.
 *
 * This class implements the builder pattern for `std::shared_ptr<ucxx::Address>`.
 * Construction happens when `build()` is called or when the builder is implicitly converted
 * to `std::shared_ptr<Address>`.
 */
class AddressBuilder final {
 public:
  /**
   * @brief Constructor for `AddressBuilder` from a worker.
   *
   * @param[in] worker worker from which to get the address.
   */
  explicit AddressBuilder(std::shared_ptr<Worker> worker);

  /**
   * @brief Constructor for `AddressBuilder` from a serialized address string.
   *
   * @param[in] addressString serialized worker address.
   */
  explicit AddressBuilder(std::string_view addressString);

  /** @brief `AddressBuilder` destructor. */
  ~AddressBuilder();

  /** @brief Copy constructor (deep-copies internal state). */
  AddressBuilder(const AddressBuilder& other);
  /** @brief Copy assignment operator (deep-copies internal state). */
  AddressBuilder& operator=(const AddressBuilder& other);
  /** @brief Move constructor. */
  AddressBuilder(AddressBuilder&&) noexcept;
  /** @brief Move assignment operator. */
  AddressBuilder& operator=(AddressBuilder&&) noexcept;

  /**
   * @brief Implicit conversion operator to `shared_ptr<Address>`.
   *
   * @return The constructed `shared_ptr<ucxx::Address>` object.
   */
  operator std::shared_ptr<Address>();

  /**
   * @brief Build and return the `Address`.
   *
   * Each call to build() creates a new `Address` instance with the current parameters.
   *
   * @return The constructed `shared_ptr<ucxx::Address>` object.
   */
  [[nodiscard]] std::shared_ptr<Address> build();

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

}  // namespace ucxx
