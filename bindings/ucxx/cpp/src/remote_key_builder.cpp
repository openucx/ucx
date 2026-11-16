/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <stdexcept>
#include <utility>

#include <internal/constructors.h>
#include <ucxx/detail/builder_utils.h>
#include <ucxx/remote_key_builder.h>

namespace ucxx {

enum class RemoteKeyBuilderSource { MemoryHandle, Serialized };

struct RemoteKeyBuilder::Impl {
  RemoteKeyBuilderSource source;
  std::shared_ptr<MemoryHandle> memoryHandle{nullptr};
  std::shared_ptr<Endpoint> endpoint{nullptr};
  SerializedRemoteKey serializedRemoteKey{};

  explicit Impl(std::shared_ptr<MemoryHandle> mh)
    : source(RemoteKeyBuilderSource::MemoryHandle), memoryHandle(std::move(mh))
  {
  }

  Impl(std::shared_ptr<Endpoint> ep, SerializedRemoteKey serialized)
    : source(RemoteKeyBuilderSource::Serialized),
      endpoint(std::move(ep)),
      serializedRemoteKey(std::move(serialized))
  {
  }
};

RemoteKeyBuilder::RemoteKeyBuilder(std::shared_ptr<MemoryHandle> memoryHandle)
  : _impl(std::make_unique<Impl>(std::move(memoryHandle)))
{
}

RemoteKeyBuilder::RemoteKeyBuilder(std::shared_ptr<Endpoint> endpoint,
                                   SerializedRemoteKey serializedRemoteKey)
  : _impl(std::make_unique<Impl>(std::move(endpoint), std::move(serializedRemoteKey)))
{
}

UCXX_BUILDER_PIMPL_DEFAULTS(RemoteKeyBuilder, RemoteKey)

std::shared_ptr<RemoteKey> RemoteKeyBuilder::build()
{
  switch (_impl->source) {
    case RemoteKeyBuilderSource::MemoryHandle:
      return detail::ConstructorFactory::createRemoteKeyFromMemoryHandle(_impl->memoryHandle);
    case RemoteKeyBuilderSource::Serialized:
      return detail::ConstructorFactory::createRemoteKeyFromSerialized(_impl->endpoint,
                                                                       _impl->serializedRemoteKey);
  }

  throw std::logic_error("Invalid RemoteKeyBuilder source");
}

}  // namespace ucxx
