/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>

#include <ucxx/component.h>

namespace ucxx {

Component::~Component() {}

// Called from child's constructor
void Component::setParent(std::shared_ptr<Component> parent) { _parent = parent; }

std::shared_ptr<Component> Component::getParent() const { return _parent; }

}  // namespace ucxx
