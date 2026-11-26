/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>

namespace ucxx {

namespace detail {
class ConstructorFactory;
}

/**
 * @brief A UCXX component class to prevent early destruction of parent object.
 *
 * A UCXX component class storing a pointer of its parent to prevent it from being
 * destroyed while child is still alive.
 */
class Component : public std::enable_shared_from_this<Component> {
 protected:
  std::shared_ptr<Component> _parent{nullptr};  ///< A reference-counted pointer to the parent.

 public:
  virtual ~Component();

  /**
   * @brief Set the internal parent reference.
   *
   * Set the internal parent reference.
   *
   * @param[in] parent the reference-counted pointer to the parent.
   */
  void setParent(std::shared_ptr<Component> parent);

  /**
   * @brief Get the internal parent reference.
   *
   * Get the internal parent reference.
   *
   * @returns the reference-counted pointer to the parent.
   */
  [[nodiscard]] std::shared_ptr<Component> getParent() const;
};

}  // namespace ucxx
