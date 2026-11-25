/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <stdexcept>
#include <utility>

#include <ucxx/buffer.h>
#include <ucxx/log.h>

#include <cuda/buffer>
#include <cuda/memory_resource>

#include <cuda_runtime_api.h>

namespace ucxx {

/**
 * @brief Concrete CCCL buffer implementation.
 */
struct CCCLBufferImpl {
  struct ReadyDevicePool : ::cuda::device_memory_pool_ref {
    explicit ReadyDevicePool(::cuda::device_memory_pool_ref pool) : device_memory_pool_ref{pool} {}

    // Buffer exposes a pointer without a consumer-stream contract. Allocate
    // synchronously on CCCL's allocation stream, not the application's default
    // stream. Return memory on the same internal stream; consumers must finish
    // accessing the buffer before it is destroyed.
    void* allocate(::cuda::stream_ref, size_t bytes, size_t alignment)
    {
      return allocate_sync(bytes, alignment);
    }

    void deallocate(::cuda::stream_ref, void* ptr, size_t bytes, size_t alignment) noexcept
    {
      deallocate_sync(ptr, bytes, alignment);
    }
  };

  using cccl_buffer_type = ::cuda::buffer<::cuda::std::byte, ::cuda::mr::device_accessible>;
  cccl_buffer_type buffer;

  // CCCL's cuda::device_default_memory_pool() requires an active CUDA primary context.
  // cudaFree(0) is the standard zero-cost idiom to initialize it.
  static auto get_device_pool()
  {
    cudaFree(0);  // Ensure CUDA primary context is initialized
    return ::cuda::device_default_memory_pool(::cuda::device_ref{0});
  }

  explicit CCCLBufferImpl(const size_t size)
    : buffer{::cudaStream_t{0}, ReadyDevicePool{get_device_pool()}, size, ::cuda::no_init}
  {
  }
};

CCCLBuffer::CCCLBuffer(const size_t size)
  : Buffer(BufferType::CCCL, size), _impl{std::make_unique<CCCLBufferImpl>(size)}
{
  ucxx_trace_data("ucxx::CCCLBuffer created: %p, impl: %p, size: %lu", this, _impl.get(), size);
}

CCCLBuffer::~CCCLBuffer() = default;

void* CCCLBuffer::data()
{
  ucxx_trace_data("ucxx::CCCLBuffer::%s, CCCLBuffer: %p, impl: %p", __func__, this, _impl.get());
  if (!_impl) throw std::runtime_error("Invalid object or already released");

  // Explicit cast required: cuda::buffer::data() returns cuda::std::byte*, not void*.
  return static_cast<void*>(_impl->buffer.data());
}

}  // namespace ucxx
