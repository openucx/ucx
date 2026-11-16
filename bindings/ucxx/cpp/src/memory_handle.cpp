/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdio>
#include <memory>
#include <string>

#include <ucp/api/ucp.h>

#include <internal/constructors.h>
#include <ucs/memory/memory_type.h>
#include <ucxx/log.h>
#include <ucxx/memory_handle.h>
#include <ucxx/utils/ucx.h>

namespace ucxx {

MemoryHandle::MemoryHandle(std::shared_ptr<Context> context,
                           const size_t size,
                           void* buffer,
                           const ucs_memory_type_t memoryType)
{
  setParent(context);

  ucp_mem_map_params_t params;
  if (buffer == nullptr) {
    params = {.field_mask = UCP_MEM_MAP_PARAM_FIELD_FLAGS | UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                            UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE,
              .length      = size,
              .flags       = UCP_MEM_MAP_NONBLOCK | UCP_MEM_MAP_ALLOCATE,
              .memory_type = memoryType};
  } else {
    params = {.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                            UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE,
              .address     = buffer,
              .length      = size,
              .memory_type = memoryType};
  }

  utils::ucsErrorThrow(ucp_mem_map(context->getHandle(), &params, &_handle));

  ucp_mem_attr_t attr = {.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS | UCP_MEM_ATTR_FIELD_LENGTH |
                                       UCP_MEM_ATTR_FIELD_MEM_TYPE};

  utils::ucsErrorThrow(ucp_mem_query(_handle, &attr));

  _baseAddress = reinterpret_cast<uint64_t>(attr.address);
  _size        = attr.length;
  _memoryType  = attr.mem_type;

  ucxx_trace("MemoryHandle created: %p, UCP handle: %p, base address: 0x%lx, size: %lu, type: %d",
             this,
             _handle,
             _baseAddress,
             _size,
             _memoryType);
}

MemoryHandle::~MemoryHandle()
{
  ucp_mem_unmap(std::static_pointer_cast<Context>(getParent())->getHandle(), _handle);
  ucxx_trace(
    "ucxx::MemoryHandle destroyed: %p, UCP handle: %p, base address: 0x%lx, size: %lu, type: %d",
    this,
    _handle,
    _baseAddress,
    _size,
    _memoryType);
}

std::shared_ptr<MemoryHandle> detail::ConstructorFactory::createMemoryHandle(
  std::shared_ptr<Context> context,
  const size_t size,
  void* buffer,
  const ucs_memory_type_t memoryType)
{
  return std::shared_ptr<MemoryHandle>(new MemoryHandle(context, size, buffer, memoryType));
}

ucp_mem_h MemoryHandle::getHandle() { return _handle; }

size_t MemoryHandle::getSize() const { return _size; }

uint64_t MemoryHandle::getBaseAddress() { return _baseAddress; }

ucs_memory_type_t MemoryHandle::getMemoryType() { return _memoryType; }

RemoteKeyBuilder MemoryHandle::remoteKeyBuilder()
{
  return RemoteKeyBuilder(std::static_pointer_cast<MemoryHandle>(shared_from_this()));
}

}  // namespace ucxx
