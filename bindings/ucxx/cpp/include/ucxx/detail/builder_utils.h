/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>

/**
 * @brief Define pImpl rule-of-five members and the implicit conversion operator for a builder.
 *
 * Place this in the `.cpp` file after the `Impl` struct definition. The builder class must
 * have a `std::unique_ptr<Impl> _impl` member and a `build()` method.
 *
 * @param BuilderClass  the fully-qualified builder class name.
 * @param TargetClass   the type that `build()` returns wrapped in `std::shared_ptr`.
 */
#define UCXX_BUILDER_PIMPL_DEFAULTS(BuilderClass, TargetClass)              \
  BuilderClass::~BuilderClass()                                  = default; \
  BuilderClass::BuilderClass(BuilderClass&&) noexcept            = default; \
  BuilderClass& BuilderClass::operator=(BuilderClass&&) noexcept = default; \
  BuilderClass::BuilderClass(const BuilderClass& other)                     \
    : _impl(std::make_unique<Impl>(*other._impl))                           \
  {                                                                         \
  }                                                                         \
  BuilderClass& BuilderClass::operator=(const BuilderClass& other)          \
  {                                                                         \
    if (this != &other) _impl = std::make_unique<Impl>(*other._impl);       \
    return *this;                                                           \
  }                                                                         \
  BuilderClass::operator std::shared_ptr<TargetClass>() { return build(); }
