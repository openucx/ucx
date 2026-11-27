/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>

#include <internal/constructors.h>
#include <ucxx/context.h>
#include <ucxx/context_builder.h>
#include <ucxx/detail/builder_utils.h>

namespace ucxx {

struct ContextBuilder::Impl {
  ConfigMap configMap{};
  uint64_t featureFlags;

  explicit Impl(uint64_t flags) : featureFlags(flags) {}
};

ContextBuilder::ContextBuilder(uint64_t featureFlags) : _impl(std::make_unique<Impl>(featureFlags))
{
}

UCXX_BUILDER_PIMPL_DEFAULTS(ContextBuilder, Context)

ContextBuilder& ContextBuilder::configMap(ConfigMap configMap)
{
  _impl->configMap = std::move(configMap);
  return *this;
}

std::shared_ptr<Context> ContextBuilder::build()
{
  return detail::ConstructorFactory::createContext(_impl->configMap, _impl->featureFlags);
}

}  // namespace ucxx
