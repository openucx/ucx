/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>

#include <internal/constructors.h>
#include <ucxx/detail/builder_utils.h>
#include <ucxx/memory_handle_builder.h>

namespace ucxx {

struct MemoryHandleBuilder::Impl {
  std::shared_ptr<Context> context{nullptr};
  size_t size{0};
  void* buffer{nullptr};
  ucs_memory_type_t memoryType{UCS_MEMORY_TYPE_HOST};

  Impl(std::shared_ptr<Context> ctx, size_t allocationSize)
    : context(std::move(ctx)), size(allocationSize)
  {
  }
};

MemoryHandleBuilder::MemoryHandleBuilder(std::shared_ptr<Context> context, size_t size)
  : _impl(std::make_unique<Impl>(std::move(context), size))
{
}

UCXX_BUILDER_PIMPL_DEFAULTS(MemoryHandleBuilder, MemoryHandle)

MemoryHandleBuilder& MemoryHandleBuilder::buffer(void* buffer)
{
  _impl->buffer = buffer;
  return *this;
}

MemoryHandleBuilder& MemoryHandleBuilder::memoryType(ucs_memory_type_t memoryType)
{
  _impl->memoryType = memoryType;
  return *this;
}

std::shared_ptr<MemoryHandle> MemoryHandleBuilder::build()
{
  return detail::ConstructorFactory::createMemoryHandle(
    _impl->context, _impl->size, _impl->buffer, _impl->memoryType);
}

}  // namespace ucxx
