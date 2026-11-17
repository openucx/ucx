/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

#include <ucxx/buffer.h>

namespace ucxx {

Buffer::Buffer(const BufferType bufferType, const size_t size)
  : _bufferType{bufferType}, _size{size}
{
}

Buffer::~Buffer() {}

BufferType Buffer::getType() const noexcept { return _bufferType; }

size_t Buffer::getSize() const noexcept { return _size; }

HostBuffer::HostBuffer(const size_t size) : Buffer(BufferType::Host, size), _buffer{malloc(size)}
{
  if (size > 0 && _buffer == nullptr) throw std::bad_alloc();
  ucxx_trace_data("ucxx::HostBuffer created: %p, buffer: %p, size: %lu", this, _buffer, size);
}

HostBuffer::HostBuffer(const void* buffer, const size_t size) : HostBuffer(size)
{
  std::memcpy(_buffer, buffer, size);
}

HostBuffer::~HostBuffer()
{
  if (_buffer) free(_buffer);
}

void* HostBuffer::release()
{
  ucxx_trace_data("ucxx::HostBuffer::%s, HostBuffer: %p, buffer: %p", __func__, this, _buffer);
  if (!_buffer) throw std::runtime_error("Invalid object or already released");

  _bufferType = ucxx::BufferType::Invalid;
  _size       = 0;

  return std::exchange(_buffer, nullptr);
}

void* HostBuffer::data()
{
  ucxx_trace_data("ucxx::HostBuffer::%s, HostBuffer: %p, buffer: %p", __func__, this, _buffer);
  if (!_buffer) throw std::runtime_error("Invalid object or already released");

  return _buffer;
}

std::shared_ptr<Buffer> allocateBuffer(const BufferType bufferType, const size_t size)
{
  if (bufferType == BufferType::CCCL) {
#if UCXX_ENABLE_CCCL
    return std::make_shared<CCCLBuffer>(size);
#else
    throw std::runtime_error("CCCL support not enabled, please compile with -DUCXX_ENABLE_CCCL=1");
#endif
  } else {
    return std::make_shared<HostBuffer>(size);
  }
}

}  // namespace ucxx
