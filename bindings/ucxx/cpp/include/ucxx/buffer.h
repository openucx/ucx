/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>

#include <ucxx/log.h>

namespace ucxx {
/**
 * @brief The type of a buffer.
 *
 * The type of a buffer that can be used to match among the different supported types.
 */
enum class BufferType {
  Host = 0,
  CCCL,
  Invalid,
};

/**
 * @brief A simple object to simplify managing buffers.
 *
 * UCXX can work with raw pointers in most cases, but in some circumstances it's required
 * to know more information about the buffer, such as with `ucxx::RequestTagMulti`. In such
 * circumstances it may need to allocate buffers internally and UCXX can utilize objects of
 * this type to describe the internally-allocated buffers.
 */
class Buffer {
 protected:
  BufferType _bufferType{BufferType::Invalid};  ///< Buffer type
  size_t _size;                                 ///< Buffer size

  /**
   * @brief Protected constructor of abstract type `Buffer`.
   *
   * This is the constructor that should be called by derived classes to store
   * general information about the buffer, such as its type and size.
   *
   * @param[in] bufferType the type of buffer the object holds.
   * @param[in] size the size of the contained buffer.
   */
  Buffer(const BufferType bufferType, const size_t size);

 public:
  Buffer()                         = delete;
  Buffer(const Buffer&)            = delete;
  Buffer& operator=(Buffer const&) = delete;
  Buffer(Buffer&& o)               = delete;
  Buffer& operator=(Buffer&& o)    = delete;

  /**
   * @brief Virtual destructor.
   *
   * Virtual destructor with empty implementation.
   */
  virtual ~Buffer();

  /**
   * @brief Get the type of buffer the object holds.
   *
   * The type of buffer the object holds is important to ensure proper casting
   * of the object into the correct derived type.
   *
   * @return the type of buffer the object holds
   */
  [[nodiscard]] BufferType getType() const noexcept;

  /**
   * @brief Get the size of the contained buffer.
   *
   * The size in bytes of the contained buffer.
   *
   * @return the size of the contained buffer.
   */
  [[nodiscard]] size_t getSize() const noexcept;

  /**
   * @brief Abstract method returning void pointer to buffer.
   *
   * Get the void pointer to the underlying buffer that holds the data. This
   * is meant to return the actual allocation, and not a pointer to some
   * container to the buffer it holds.
   *
   * @return the void pointer to the buffer.
   */
  [[nodiscard]] virtual void* data() = 0;
};

/**
 * @brief A simple object containing a host buffer.
 *
 * A buffer encapsulating a host buffer with its properties.
 */
class HostBuffer : public Buffer {
 private:
  void* _buffer;  ///< Pointer to the allocated buffer

 public:
  HostBuffer()                             = delete;
  HostBuffer(const HostBuffer&)            = delete;
  HostBuffer& operator=(HostBuffer const&) = delete;
  HostBuffer(HostBuffer&& o)               = delete;
  HostBuffer& operator=(HostBuffer&& o)    = delete;

  /**
   * @brief Constructor of concrete type `HostBuffer`.
   *
   * Constructor to materialize a buffer holding host memory. The internal buffer
   * is allocated using `malloc`, and thus should be freed with `free`.
   *
   * @param[in] size the size of the host buffer to allocate.
   *
   * @code{.cpp}
   * // Allocate host buffer of 1KiB
   * auto buffer = HostBuffer(1024);
   * @endcode
   */
  explicit HostBuffer(const size_t size);

  /**
   * @brief Construct a host buffer by deep copying the contents of another buffer.
   *
   * @param[in] buffer the data to copy from.
   * @param[in] size the size of the host buffer to allocate.
   *
   * @code{.cpp}
   * auto buffer_copy = HostBuffer(buffer.data(), buffer.getSize());
   * @endcode
   */
  HostBuffer(const void* buffer, const size_t size);

  /**
   * @brief Destructor of concrete type `HostBuffer`.
   *
   * Frees the underlying buffer, unless the underlying buffer was released to
   * the user after a call to `release`.
   */
  ~HostBuffer();

  /**
   * @brief Release the allocated host buffer to the caller.
   *
   * Release ownership of the buffer to the caller. After this method is called,
   * the caller becomes responsible for its deallocation once it is not needed
   * anymore. The buffer is allocated with `malloc`, and should be properly
   * disposed of by a call to `free`.
   *
   * The original `HostBuffer` object becomes invalid.
   *
   * @code{.cpp}
   * // Allocate host buffer of 1KiB
   * auto buffer = HostBuffer(1024);
   * void* bufferPtr = buffer.release();
   *
   * // do work on bufferPtr
   *
   * // Free buffer
   * free(bufferPtr);
   * @endcode
   *
   * @throws std::runtime_error if object has been released.
   *
   * @return the void pointer to the buffer.
   */
  [[nodiscard]] void* release();

  /**
   * @brief Get a pointer to the allocated raw host buffer.
   *
   * Get a pointer to the underlying buffer, but does not release ownership.
   *
   * @code{.cpp}
   * // Allocate host buffer of 1KiB
   * auto buffer = HostBuffer(1024);
   * void* bufferPtr = buffer.data();
   *
   * // do work on bufferPtr
   *
   * // Memory is freed once `buffer` goes out-of-scope.
   * @endcode
   *
   * @throws std::runtime_error if object has been released.
   *
   * @return the void pointer to the buffer.
   */
  [[nodiscard]] void* data() override;
};

#if UCXX_ENABLE_CCCL
/**
 * @brief Opaque implementation struct for CCCLBuffer (defined in buffer_cccl.cu).
 * Users should NOT access this type directly.
 */
struct CCCLBufferImpl;

/**
 * @brief A simple object containing a CCCL (CUDA) buffer.
 *
 * A buffer encapsulating a CCCL (CUDA) buffer with its properties.
 */
class CCCLBuffer : public Buffer {
 private:
  std::unique_ptr<CCCLBufferImpl> _impl;  ///< Opaque unique_ptr to CCCL buffer implementation

 public:
  CCCLBuffer()                             = delete;
  CCCLBuffer(const CCCLBuffer&)            = delete;
  CCCLBuffer& operator=(CCCLBuffer const&) = delete;
  CCCLBuffer(CCCLBuffer&& o)               = delete;
  CCCLBuffer& operator=(CCCLBuffer&& o)    = delete;

  ~CCCLBuffer() override;

  /**
   * @brief Constructor of concrete type `CCCLBuffer`.
   *
   * Constructor to materialize a buffer holding device memory. The internal
   * buffer holds a `std::unique_ptr<CCCLBufferImpl>` and is destroyed
   * when the object goes out-of-scope or is explicitly deleted.
   *
   * @param[in] size the size of the device buffer to allocate.
   *
   * @code{.cpp}
   * // Allocate CCCL device buffer of 1KiB
   * auto buffer = CCCLBuffer(1024);
   * @endcode
   */
  explicit CCCLBuffer(const size_t size);

  /**
   * @brief Get a pointer to the allocated raw device buffer.
   *
   * Get a pointer to the underlying buffer, but does not release ownership.
   *
   * @code{.cpp}
   * auto buffer = CCCLBuffer(1024);
   * void* ptr = buffer.data();
   * // `ptr` is a device pointer valid until buffer is released/destroyed
   * @endcode
   *
   * @throws std::runtime_error if object has been released.
   *
   * @return the void pointer to the device buffer.
   */
  [[nodiscard]] void* data() override;
};
#endif

/**
 * @brief Allocate a buffer of specified type and size.
 *
 * Allocate a buffer of the specified type and size pair, returning the `ucxx::Buffer`
 * object wrapped in a `std::shared_ptr`.
 *
 * @param[in] bufferType  the type of buffer to allocate.
 * @param[in] size        the size (in bytes) of the buffer to allocate.
 *
 * @returns the `std::shared_ptr` to the allocated buffer.
 */
[[nodiscard]] std::shared_ptr<Buffer> allocateBuffer(BufferType bufferType, const size_t size);

}  // namespace ucxx
