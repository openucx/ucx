/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <ucxx/api.h>

#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef UCXX_BENCHMARKS_ENABLE_RMM
#include <cuda/stream>
#include <rmm/aligned.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>
#endif

#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
// CUDA error checking macro (if not already defined)
#ifndef CUDA_EXIT_ON_ERROR
#define CUDA_EXIT_ON_ERROR(operation, context)                                                  \
  ([&]() {                                                                                      \
    cudaError_t err = operation;                                                                \
    if (err != cudaSuccess) {                                                                   \
      std::cerr << "CUDA error in " << context << " at " << __FILE__ << ":" << __LINE__ << ": " \
                << cudaGetErrorString(err) << std::endl;                                        \
      std::exit(-1);                                                                            \
    }                                                                                           \
    return err;                                                                                 \
  })()
#endif
#endif

enum class MemoryType {
  Host,
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
  Cuda,
  CudaManaged,
  CudaAsync,
#endif
#ifdef UCXX_BENCHMARKS_ENABLE_RMM
  RmmDeviceMemoryResource,
  RmmPoolMemoryResource,
  RmmCudaAsyncMemoryResource,
  RmmCudaAsyncManagedMemoryResource,
#endif
#ifdef UCXX_BENCHMARKS_ENABLE_CCCL
  CcclDeviceMemoryPool,
  CcclSharedResource,
  CcclCudaAsyncResource,
  CcclCudaAsyncManagedResource,
#endif
};

/**
 * @brief Get the string representation of the memory type
 *
 * Converts a MemoryType enum value to its corresponding lowercase string.
 *
 * @param memoryType The memory type
 * @return The string representation of the memory type
 *
 * @throws std::runtime_error if the memory type is unknown
 */
std::string getMemoryTypeString(MemoryType memoryType) {
  if (memoryType == MemoryType::Host)
    return "host";
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
  else if (memoryType == MemoryType::Cuda)
    return "cuda";
  else if (memoryType == MemoryType::CudaManaged)
    return "cuda-managed";
  else if (memoryType == MemoryType::CudaAsync)
    return "cuda-async";
#endif
#ifdef UCXX_BENCHMARKS_ENABLE_RMM
  else if (memoryType == MemoryType::RmmDeviceMemoryResource)
    return "rmm-cuda";
  else if (memoryType == MemoryType::RmmPoolMemoryResource)
    return "rmm-pool";
  else if (memoryType == MemoryType::RmmCudaAsyncMemoryResource)
    return "rmm-cuda-async";
  else if (memoryType == MemoryType::RmmCudaAsyncManagedMemoryResource)
    return "rmm-cuda-async-managed";
#endif
#ifdef UCXX_BENCHMARKS_ENABLE_CCCL
  else if (memoryType == MemoryType::CcclDeviceMemoryPool)
    return "cccl-device";
  else if (memoryType == MemoryType::CcclSharedResource)
    return "cccl-shared";
  else if (memoryType == MemoryType::CcclCudaAsyncResource)
    return "cccl-cuda-async";
  else if (memoryType == MemoryType::CcclCudaAsyncManagedResource)
    return "cccl-cuda-async-managed";
#endif
  else {
    throw std::runtime_error("Unknown memory type");
  }
}

/**
 * @brief Get the memory type from a string
 *
 * Converts a string to the corresponding MemoryType enum value.
 *
 * @param memoryTypeString String representation of the memory type (e.g., "host", "cuda",
 * "cuda-managed", "cuda-async").
 * @return MemoryType corresponding to the input string.
 *
 * @throws std::runtime_error if the string does not match a known memory type.
 */
MemoryType getMemoryTypeFromString(std::string memoryTypeString) {
  if (memoryTypeString == "host")
    return MemoryType::Host;
#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
  else if (memoryTypeString == "cuda")
    return MemoryType::Cuda;
  else if (memoryTypeString == "cuda-managed")
    return MemoryType::CudaManaged;
  else if (memoryTypeString == "cuda-async")
    return MemoryType::CudaAsync;
#endif
#ifdef UCXX_BENCHMARKS_ENABLE_RMM
  else if (memoryTypeString == "rmm-cuda")
    return MemoryType::RmmDeviceMemoryResource;
  else if (memoryTypeString == "rmm-pool")
    return MemoryType::RmmPoolMemoryResource;
  else if (memoryTypeString == "rmm-cuda-async")
    return MemoryType::RmmCudaAsyncMemoryResource;
  else if (memoryTypeString == "rmm-cuda-async-managed")
    return MemoryType::RmmCudaAsyncManagedMemoryResource;
#endif
#ifdef UCXX_BENCHMARKS_ENABLE_CCCL
  else if (memoryTypeString == "cccl-device")
    return MemoryType::CcclDeviceMemoryPool;
  else if (memoryTypeString == "cccl-shared")
    return MemoryType::CcclSharedResource;
  else if (memoryTypeString == "cccl-cuda-async")
    return MemoryType::CcclCudaAsyncResource;
  else if (memoryTypeString == "cccl-cuda-async-managed")
    return MemoryType::CcclCudaAsyncManagedResource;
#endif
  else {
    throw std::runtime_error("Unknown memory type");
  }
}

#ifdef UCXX_BENCHMARKS_ENABLE_RMM
/**
 * @brief Returns true if \p t selects an RMM memory resource allocation path.
 */
inline bool isRmmMemoryType(MemoryType t)
{
  return t == MemoryType::RmmDeviceMemoryResource || t == MemoryType::RmmPoolMemoryResource ||
         t == MemoryType::RmmCudaAsyncMemoryResource || t == MemoryType::RmmCudaAsyncManagedMemoryResource;
}
#endif

#ifdef UCXX_BENCHMARKS_ENABLE_CCCL
/**
 * @brief Returns true if \p t selects a CCCL buffer allocation path.
 */
inline bool isCcclMemoryType(MemoryType t)
{
  return t == MemoryType::CcclDeviceMemoryPool || t == MemoryType::CcclSharedResource ||
         t == MemoryType::CcclCudaAsyncResource || t == MemoryType::CcclCudaAsyncManagedResource;
}
#endif

enum class DirectionType { Send, Recv };

typedef std::unordered_map<DirectionType, std::vector<char>> BufferMap;
typedef std::shared_ptr<BufferMap> BufferMapPtr;

#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
// CUDA memory buffer structure (conditional)
struct CudaBuffer {
  void* ptr{nullptr};
  size_t size{0};

  CudaBuffer() = default;
  explicit CudaBuffer(size_t bufferSize) : size(bufferSize)
  {
    CUDA_EXIT_ON_ERROR(cudaMalloc(&ptr, size), "CUDA memory allocation");
  }

  ~CudaBuffer()
  {
    if (ptr) { CUDA_EXIT_ON_ERROR(cudaFree(ptr), "CUDA memory deallocation"); }
  }

  CudaBuffer(const CudaBuffer&)            = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;
  CudaBuffer(CudaBuffer&& other) noexcept : ptr(other.ptr), size(other.size)
  {
    other.ptr  = nullptr;
    other.size = 0;
  }
  CudaBuffer& operator=(CudaBuffer&& other) noexcept
  {
    if (this != &other) {
      if (ptr) CUDA_EXIT_ON_ERROR(cudaFree(ptr), "CUDA memory deallocation");
      ptr        = other.ptr;
      size       = other.size;
      other.ptr  = nullptr;
      other.size = 0;
    }
    return *this;
  }
};

// CUDA managed memory buffer structure
struct CudaManagedBuffer {
  void* ptr{nullptr};
  size_t size{0};

  CudaManagedBuffer() = default;
  explicit CudaManagedBuffer(size_t bufferSize) : size(bufferSize)
  {
    CUDA_EXIT_ON_ERROR(cudaMallocManaged(&ptr, size), "CUDA managed memory allocation");
  }

  ~CudaManagedBuffer()
  {
    if (ptr) { CUDA_EXIT_ON_ERROR(cudaFree(ptr), "CUDA managed memory deallocation"); }
  }

  CudaManagedBuffer(const CudaManagedBuffer&)            = delete;
  CudaManagedBuffer& operator=(const CudaManagedBuffer&) = delete;
  CudaManagedBuffer(CudaManagedBuffer&& other) noexcept : ptr(other.ptr), size(other.size)
  {
    other.ptr  = nullptr;
    other.size = 0;
  }
  CudaManagedBuffer& operator=(CudaManagedBuffer&& other) noexcept
  {
    if (this != &other) {
      if (ptr) CUDA_EXIT_ON_ERROR(cudaFree(ptr), "CUDA managed memory deallocation");
      ptr        = other.ptr;
      size       = other.size;
      other.ptr  = nullptr;
      other.size = 0;
    }
    return *this;
  }
};

// CUDA async memory buffer structure
struct CudaAsyncBuffer {
  void* ptr{nullptr};
  size_t size{0};
  cudaStream_t stream{nullptr};

  CudaAsyncBuffer() = default;
  explicit CudaAsyncBuffer(size_t bufferSize) : size(bufferSize)
  {
    CUDA_EXIT_ON_ERROR(cudaStreamCreate(&stream), "CUDA stream creation");
    CUDA_EXIT_ON_ERROR(cudaMallocAsync(&ptr, size, stream), "CUDA async memory allocation");
  }

  ~CudaAsyncBuffer()
  {
    if (ptr) { CUDA_EXIT_ON_ERROR(cudaFreeAsync(ptr, stream), "CUDA async memory deallocation"); }
    if (stream) { CUDA_EXIT_ON_ERROR(cudaStreamDestroy(stream), "CUDA stream destruction"); }
  }

  CudaAsyncBuffer(const CudaAsyncBuffer&)            = delete;
  CudaAsyncBuffer& operator=(const CudaAsyncBuffer&) = delete;
  CudaAsyncBuffer(CudaAsyncBuffer&& other) noexcept
    : ptr(other.ptr), size(other.size), stream(other.stream)
  {
    other.ptr    = nullptr;
    other.size   = 0;
    other.stream = nullptr;
  }
  /**
   * @brief Move assignment operator for CudaAsyncBuffer.
   *
   * Releases any existing CUDA asynchronous memory and stream resources, then transfers ownership
   * of the memory pointer, size, and stream from another CudaAsyncBuffer instance. The source
   * instance is reset to a null state.
   *
   * @param other The CudaAsyncBuffer to move from.
   * @return Reference to this CudaAsyncBuffer.
   */
  CudaAsyncBuffer& operator=(CudaAsyncBuffer&& other) noexcept
  {
    if (this != &other) {
      if (ptr) CUDA_EXIT_ON_ERROR(cudaFreeAsync(ptr, stream), "CUDA async memory deallocation");
      if (stream) CUDA_EXIT_ON_ERROR(cudaStreamDestroy(stream), "CUDA stream destruction");
      ptr          = other.ptr;
      size         = other.size;
      stream       = other.stream;
      other.ptr    = nullptr;
      other.size   = 0;
      other.stream = nullptr;
    }
    return *this;
  }
};

typedef std::unordered_map<DirectionType, CudaBuffer> CudaBufferMap;
typedef std::unordered_map<DirectionType, CudaManagedBuffer> CudaManagedBufferMap;
typedef std::unordered_map<DirectionType, CudaAsyncBuffer> CudaAsyncBufferMap;
typedef std::shared_ptr<CudaBufferMap> CudaBufferMapPtr;
typedef std::shared_ptr<CudaManagedBufferMap> CudaManagedBufferMapPtr;
typedef std::shared_ptr<CudaAsyncBufferMap> CudaAsyncBufferMapPtr;
#endif

typedef std::unordered_map<DirectionType, ucxx::Tag> TagMap;
typedef std::shared_ptr<TagMap> TagMapPtr;

// Forward declaration of allocation function
BufferMapPtr allocateTransferBuffers(size_t messageSize);

// Unified buffer interface for different memory types
struct BufferInterface {
  virtual void* getSendPtr()                      = 0;
  virtual void* getRecvPtr()                      = 0;
  virtual void verifyResults(size_t messageSize) = 0;
  virtual ~BufferInterface()                      = default;
};

// Host memory buffer implementation
struct HostBufferInterface : public BufferInterface {
  BufferMapPtr bufferMap;

  /**
   * @brief Constructs a HostBufferInterface with the provided buffer map.
   *
   * @param buf Buffer map containing send and receive host buffers.
   */
  explicit HostBufferInterface(BufferMapPtr buf) : bufferMap(buf) {}

  /**
   * @brief Returns a pointer to the send buffer data.
   *
   * @return void* Pointer to the start of the send buffer.
   */
  void* getSendPtr() override { return (*bufferMap)[DirectionType::Send].data(); }
  /**
   * @brief Returns a pointer to the receive buffer.
   *
   * @return void* Pointer to the start of the receive buffer.
   */
  void* getRecvPtr() override { return (*bufferMap)[DirectionType::Recv].data(); }

  /**
   * @brief Verifies that the receive buffer matches the send buffer.
   *
   * Asserts that each byte in the receive buffer is equal to the corresponding byte in the send
   * buffer for the given message size.
   */
  void verifyResults(size_t messageSize) override
  {
    for (size_t j = 0; j < (*bufferMap)[DirectionType::Send].size(); ++j)
      if ((*bufferMap)[DirectionType::Recv][j] != (*bufferMap)[DirectionType::Send][j])
        throw std::runtime_error("Host data verification failed at byte " + std::to_string(j));
  }

  // Static factory method to create host buffer interface
  static std::unique_ptr<HostBufferInterface> createBufferInterface(size_t messageSize,
                                                                    bool reuseAlloc)
  {
    static BufferMapPtr reuseBuffer;

    if (reuseAlloc) {
      if (!reuseBuffer) { reuseBuffer = allocateTransferBuffers(messageSize); }
      return std::make_unique<HostBufferInterface>(reuseBuffer);
    } else {
      auto bufferMap = allocateTransferBuffers(messageSize);
      return std::make_unique<HostBufferInterface>(bufferMap);
    }
  }
};

#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
// Unified CUDA buffer interface that all CUDA buffer types implement
struct CudaBufferInterfaceBase : public BufferInterface {
  /**
   * @brief Virtual destructor for CudaBufferInterfaceBase.
   *
   * Ensures proper cleanup of derived CUDA buffer interface objects.
   */
  virtual ~CudaBufferInterfaceBase() = default;

  // Pure virtual methods that derived classes must implement
  virtual void* getBufferPtr(DirectionType direction)                         = 0;
  virtual std::shared_ptr<void> getBufferMap()                                = 0;
  virtual void initializeBuffer(DirectionType direction, size_t messageSize) = 0;
  virtual void verifyBufferResults(size_t messageSize)                       = 0;

  /**
   * @brief Returns a pointer to the send buffer.
   *
   * @return void* Pointer to the start of the buffer used for sending data.
   */
  void* getSendPtr() override { return getBufferPtr(DirectionType::Send); }
  /**
   * @brief Returns a pointer to the receive buffer.
   *
   * @return void* Pointer to the start of the buffer used for receiving data.
   */
  void* getRecvPtr() override { return getBufferPtr(DirectionType::Recv); }

  /**
   * @brief Verifies that the receive buffer matches the send buffer for the given message size.
   *
   * Delegates verification to the implementation-specific buffer result check.
   *
   * @param messageSize The number of bytes to verify.
   */
  void verifyResults(size_t messageSize) override { verifyBufferResults(messageSize); }

  // Static factory method to create appropriate buffer interface
  static std::unique_ptr<CudaBufferInterfaceBase> createBufferInterface(MemoryType memoryType,
                                                                        size_t messageSize,
                                                                        bool reuseAlloc);

  // Static allocation method that each derived class must implement
  static std::unique_ptr<CudaBufferInterfaceBase> allocateBuffers(MemoryType memoryType,
                                                                  size_t messageSize);

  // Virtual clone method for reuse buffers
  virtual std::unique_ptr<CudaBufferInterfaceBase> clone() const = 0;
};

// Template-based CUDA buffer interface implementation
template <typename BufferType>
struct CudaBufferInterface : public CudaBufferInterfaceBase {
  std::shared_ptr<std::unordered_map<DirectionType, BufferType>> bufferMap;

  /**
   * @brief Constructs a CUDA buffer interface with the given buffer map.
   *
   * @param buf Map of DirectionType to CUDA buffer objects.
   */
  explicit CudaBufferInterface(std::shared_ptr<std::unordered_map<DirectionType, BufferType>> buf)
    : bufferMap(buf)
  {
  }

  /**
   * @brief Returns a raw pointer to the buffer for the specified direction.
   *
   * @param direction The data transfer direction (send or receive).
   * @return void* Pointer to the start of the buffer corresponding to the given direction.
   */
  void* getBufferPtr(DirectionType direction) override { return (*bufferMap)[direction].ptr; }

  /**
   * @brief Returns the underlying buffer map.
   *
   * @return std::shared_ptr<void> Buffer map containing buffers for each
   * direction.
   */
  std::shared_ptr<void> getBufferMap() override { return bufferMap; }

  /**
   * @brief Initializes the send buffer with a fixed pattern.
   *
   * For the specified direction, if it is `Send`, fills the buffer with the byte pattern `0xaa`
   * using the appropriate memory copy method for the buffer type.
   *
   * @param direction The direction of the buffer to initialize. Only `DirectionType::Send` is
   * affected.
   * @param messageSize The size of the buffer to initialize, in bytes.
   */
  void initializeBuffer(DirectionType direction, size_t messageSize) override
  {
    if (direction == DirectionType::Send) {
      std::vector<char> pattern(messageSize, 0xaa);
      if constexpr (std::is_same_v<BufferType, CudaAsyncBuffer>) {
        CUDA_EXIT_ON_ERROR(cudaMemcpyAsync((*bufferMap)[DirectionType::Send].ptr,
                                           pattern.data(),
                                           messageSize,
                                           cudaMemcpyHostToDevice,
                                           (*bufferMap)[DirectionType::Send].stream),
                           "CUDA async send buffer initialization");
        CUDA_EXIT_ON_ERROR(cudaStreamSynchronize((*bufferMap)[DirectionType::Send].stream),
                           "CUDA stream synchronization");
      } else if constexpr (std::is_same_v<BufferType, CudaManagedBuffer>) {
        std::memcpy((*bufferMap)[DirectionType::Send].ptr, pattern.data(), messageSize);
      } else {
        CUDA_EXIT_ON_ERROR(
          cudaMemcpy((*bufferMap)[DirectionType::Send].ptr, pattern.data(), messageSize, cudaMemcpyHostToDevice),
          "CUDA send buffer initialization");
      }
    }
  }

  /**
   * @brief Verifies that the receive buffer matches the send buffer for the specified message
   * size.
   *
   * Copies data from device or managed memory to host memory as appropriate for the buffer type,
   * then asserts that the receive buffer contents are identical to the send buffer.
   *
   * @param messageSize Number of bytes to verify.
   */
  void verifyBufferResults(size_t messageSize) override
  {
    std::vector<char> sendData(messageSize);
    std::vector<char> recvData(messageSize);

    if constexpr (std::is_same_v<BufferType, CudaAsyncBuffer>) {
      // For async buffers, use the stream from the buffer
      CUDA_EXIT_ON_ERROR(
        cudaMemcpyAsync(sendData.data(), (*bufferMap)[DirectionType::Send].ptr, messageSize, cudaMemcpyDeviceToHost,
                       (*bufferMap)[DirectionType::Send].stream),
        "CUDA async send data copy for verification");
      CUDA_EXIT_ON_ERROR(
        cudaMemcpyAsync(recvData.data(), (*bufferMap)[DirectionType::Recv].ptr, messageSize, cudaMemcpyDeviceToHost,
                       (*bufferMap)[DirectionType::Recv].stream),
        "CUDA async recv data copy for verification");

      // Synchronize streams
      CUDA_EXIT_ON_ERROR(cudaStreamSynchronize((*bufferMap)[DirectionType::Send].stream),
                         "CUDA send stream synchronization for verification");
      CUDA_EXIT_ON_ERROR(cudaStreamSynchronize((*bufferMap)[DirectionType::Recv].stream),
                         "CUDA recv stream synchronization for verification");
    } else if constexpr (std::is_same_v<BufferType, CudaManagedBuffer>) {
      // Managed memory can be accessed directly from host
      std::memcpy(sendData.data(), (*bufferMap)[DirectionType::Send].ptr, messageSize);
      std::memcpy(recvData.data(), (*bufferMap)[DirectionType::Recv].ptr, messageSize);
    } else {
      // For non-async buffers, use synchronous copy with null stream
      CUDA_EXIT_ON_ERROR(
        cudaMemcpyAsync(sendData.data(), (*bufferMap)[DirectionType::Send].ptr, messageSize, cudaMemcpyDeviceToHost,
                       nullptr),
        "CUDA send data copy for verification");
      CUDA_EXIT_ON_ERROR(
        cudaMemcpyAsync(recvData.data(), (*bufferMap)[DirectionType::Recv].ptr, messageSize, cudaMemcpyDeviceToHost,
                       nullptr),
        "CUDA recv data copy for verification");
    }

    for (size_t j = 0; j < sendData.size(); ++j)
      if (recvData[j] != sendData[j])
        throw std::runtime_error("CUDA data verification failed at byte " + std::to_string(j));
  }

  // Static allocation methods
  static std::shared_ptr<std::unordered_map<DirectionType, BufferType>> allocateBuffers(size_t messageSize);

  /**
   * @brief Creates a new buffer interface instance sharing the same buffer map.
   *
   * @return Cloned CudaBufferInterface with shared buffer data.
   */
  std::unique_ptr<CudaBufferInterfaceBase> clone() const override
  {
    return std::make_unique<CudaBufferInterface<BufferType>>(bufferMap);
  }
};
#endif

/**
 * @brief Allocates and initializes host memory buffers for send and receive operations.
 *
 * Creates a buffer map containing two vectors of bytes: one for sending, initialized with the
 * pattern `0xaa`, and one for receiving, zero-initialized. The map is keyed by
 * `DirectionType::Send` and `DirectionType::Recv`.
 *
 * @param messageSize The size of each buffer in bytes.
 * @return BufferMapPtr The allocated buffer map.
 */
BufferMapPtr allocateTransferBuffers(size_t messageSize)
{
  return std::make_shared<BufferMap>(BufferMap{{DirectionType::Send, std::vector<char>(messageSize, 0xaa)},
                                               {DirectionType::Recv, std::vector<char>(messageSize)}});
}

#ifdef UCXX_BENCHMARKS_ENABLE_CUDA
// Type aliases for convenience
using CudaDeviceBufferInterface  = CudaBufferInterface<CudaBuffer>;
using CudaManagedBufferInterface = CudaBufferInterface<CudaManagedBuffer>;
using CudaAsyncBufferInterface   = CudaBufferInterface<CudaAsyncBuffer>;

// Template specialization implementations for allocation methods
template <>
/**
 * @brief Allocates CUDA device buffers for send and receive operations and initializes the send
 * buffer.
 *
 * Allocates two CUDA device buffers of the specified size for send and receive directions. The
 * send buffer is initialized with the byte pattern 0xaa.
 *
 * @param messageSize Size of each buffer in bytes.
 * @return std::shared_ptr<CudaBufferMap> Map containing the allocated CUDA device buffers.
 */
std::shared_ptr<CudaBufferMap> CudaBufferInterface<CudaBuffer>::allocateBuffers(size_t messageSize)
{
  auto bufferMap                    = std::make_shared<CudaBufferMap>();
  (*bufferMap)[DirectionType::Send] = CudaBuffer(messageSize);
  (*bufferMap)[DirectionType::Recv] = CudaBuffer(messageSize);

  // Initialize send buffer with pattern
  std::vector<char> pattern(messageSize, 0xaa);
  CUDA_EXIT_ON_ERROR(
    cudaMemcpy((*bufferMap)[DirectionType::Send].ptr, pattern.data(), messageSize, cudaMemcpyHostToDevice),
    "CUDA send buffer initialization");

  return bufferMap;
}

template <>
/**
 * @brief Allocates and initializes CUDA managed memory buffers for send and receive operations.
 *
 * Allocates two `CudaManagedBuffer` instances of the specified size for the send and receive
 * directions. The send buffer is initialized with the byte pattern `0xaa`.
 *
 * @param messageSize The size in bytes for each buffer.
 * @return Map associating each direction with its corresponding managed buffer.
 */
std::shared_ptr<CudaManagedBufferMap> CudaBufferInterface<CudaManagedBuffer>::allocateBuffers(
  size_t messageSize)
{
  auto bufferMap                    = std::make_shared<CudaManagedBufferMap>();
  (*bufferMap)[DirectionType::Send] = CudaManagedBuffer(messageSize);
  (*bufferMap)[DirectionType::Recv] = CudaManagedBuffer(messageSize);

  // Initialize send buffer with pattern (managed memory can be accessed from host)
  std::vector<char> pattern(messageSize, 0xaa);
  std::memcpy((*bufferMap)[DirectionType::Send].ptr, pattern.data(), messageSize);

  return bufferMap;
}

template <>
/**
 * @brief Allocates and initializes CUDA asynchronous send and receive buffers.
 *
 * Allocates device memory for send and receive buffers using CUDA asynchronous allocation, and
 * initializes the send buffer with a fixed pattern (0xaa) via asynchronous memory copy. The CUDA
 * stream is synchronized to ensure initialization is complete before use.
 *
 * @param messageSize Size in bytes for each buffer.
 * @return Map containing the allocated and initialized CUDA asynchronous buffers for send and
 * receive directions.
 */
std::shared_ptr<CudaAsyncBufferMap> CudaBufferInterface<CudaAsyncBuffer>::allocateBuffers(
  size_t messageSize)
{
  auto bufferMap                    = std::make_shared<CudaAsyncBufferMap>();
  (*bufferMap)[DirectionType::Send] = CudaAsyncBuffer(messageSize);
  (*bufferMap)[DirectionType::Recv] = CudaAsyncBuffer(messageSize);

  // Initialize send buffer with pattern using async copy
  std::vector<char> pattern(messageSize, 0xaa);
  CUDA_EXIT_ON_ERROR(cudaMemcpyAsync((*bufferMap)[DirectionType::Send].ptr,
                                     pattern.data(),
                                     messageSize,
                                     cudaMemcpyHostToDevice,
                                     (*bufferMap)[DirectionType::Send].stream),
                     "CUDA async send buffer initialization");

  // Synchronize the stream to ensure the copy is complete
  CUDA_EXIT_ON_ERROR(cudaStreamSynchronize((*bufferMap)[DirectionType::Send].stream),
                     "CUDA stream synchronization");

  return bufferMap;
}

// Factory method implementation
std::unique_ptr<CudaBufferInterfaceBase> CudaBufferInterfaceBase::createBufferInterface(
  MemoryType memoryType, size_t messageSize, bool reuseAlloc)
{
  if (reuseAlloc) {
    // Use static reuse buffers for each memory type
    static std::unordered_map<MemoryType, std::unique_ptr<CudaBufferInterfaceBase>> reuseBuffers;

    // Check if we already have a reuse buffer for this memory type
    auto it = reuseBuffers.find(memoryType);
    if (it == reuseBuffers.end()) {
      // Create new reuse buffer
      reuseBuffers[memoryType] = allocateBuffers(memoryType, messageSize);
    }

    // Clone the reuse buffer (we need to implement a clone method)
    return reuseBuffers[memoryType]->clone();
  } else {
    // Create new buffer
    return allocateBuffers(memoryType, messageSize);
  }
}

// Unified allocation method implementation
std::unique_ptr<CudaBufferInterfaceBase> CudaBufferInterfaceBase::allocateBuffers(
  MemoryType memoryType, size_t messageSize)
{
  switch (memoryType) {
    case MemoryType::Cuda: {
      auto bufferMap = CudaDeviceBufferInterface::allocateBuffers(messageSize);
      return std::make_unique<CudaDeviceBufferInterface>(bufferMap);
    }
    case MemoryType::CudaManaged: {
      auto bufferMap = CudaManagedBufferInterface::allocateBuffers(messageSize);
      return std::make_unique<CudaManagedBufferInterface>(bufferMap);
    }
    case MemoryType::CudaAsync: {
      auto bufferMap = CudaAsyncBufferInterface::allocateBuffers(messageSize);
      return std::make_unique<CudaAsyncBufferInterface>(bufferMap);
    }
    default: throw std::runtime_error("Unsupported CUDA memory type");
  }
}
#endif

#ifdef UCXX_BENCHMARKS_ENABLE_RMM

namespace rmm_benchmark_detail {

inline std::size_t initial_pool_size(std::size_t messageSize)
{
  constexpr std::size_t kMinPool = 1ull << 20;
  const std::size_t scaled       = (std::max)(messageSize * 4, kMinPool);
  return rmm::align_up(scaled, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

inline void initialize_send_pattern(rmm::device_buffer& send, std::size_t messageSize)
{
  std::vector<char> pattern(messageSize, 0xaa);
  CUDA_EXIT_ON_ERROR(cudaMemcpyAsync(send.data(),
                                     pattern.data(),
                                     messageSize,
                                     cudaMemcpyHostToDevice,
                                     send.stream().get()),
                     "RMM send buffer initialization");
  CUDA_EXIT_ON_ERROR(cudaStreamSynchronize(send.stream().get()), "RMM stream synchronization");
}

inline void verify_device_buffers(rmm::device_buffer& send,
                                  rmm::device_buffer& recv,
                                  std::size_t messageSize)
{
  std::vector<char> sendData(messageSize);
  std::vector<char> recvData(messageSize);
  CUDA_EXIT_ON_ERROR(cudaMemcpyAsync(sendData.data(),
                                     send.data(),
                                     messageSize,
                                     cudaMemcpyDeviceToHost,
                                     send.stream().get()),
                     "RMM send data copy for verification");
  CUDA_EXIT_ON_ERROR(cudaMemcpyAsync(recvData.data(),
                                     recv.data(),
                                     messageSize,
                                     cudaMemcpyDeviceToHost,
                                     recv.stream().get()),
                     "RMM recv data copy for verification");
  CUDA_EXIT_ON_ERROR(cudaStreamSynchronize(send.stream().get()),
                     "RMM send stream synchronization for verification");
  CUDA_EXIT_ON_ERROR(cudaStreamSynchronize(recv.stream().get()),
                     "RMM recv stream synchronization for verification");
  for (std::size_t j = 0; j < messageSize; ++j)
    if (recvData[j] != sendData[j])
      throw std::runtime_error("RMM data verification failed at byte " + std::to_string(j));
}

struct RmmDeviceMrBuffers {
  rmm::mr::cuda_memory_resource mr{};
  std::unique_ptr<rmm::device_buffer> send;
  std::unique_ptr<rmm::device_buffer> recv;

  static std::shared_ptr<RmmDeviceMrBuffers> allocate(std::size_t messageSize)
  {
    auto p    = std::make_shared<RmmDeviceMrBuffers>();
    p->send   = std::make_unique<rmm::device_buffer>(messageSize, cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p->mr);
    p->recv   = std::make_unique<rmm::device_buffer>(messageSize, cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p->mr);
    initialize_send_pattern(*p->send, messageSize);
    return p;
  }
};

struct RmmPoolBuffers {
  rmm::mr::pool_memory_resource pool;
  std::unique_ptr<rmm::device_buffer> send;
  std::unique_ptr<rmm::device_buffer> recv;

  explicit RmmPoolBuffers(std::size_t messageSize)
    : pool(rmm::mr::cuda_memory_resource{}, rmm_benchmark_detail::initial_pool_size(messageSize))
  {
    send     = std::make_unique<rmm::device_buffer>(messageSize, cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, pool);
    recv     = std::make_unique<rmm::device_buffer>(messageSize, cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, pool);
    rmm_benchmark_detail::initialize_send_pattern(*send, messageSize);
  }

  static std::shared_ptr<RmmPoolBuffers> allocate(std::size_t messageSize)
  {
    return std::make_shared<RmmPoolBuffers>(messageSize);
  }
};

struct RmmCudaAsyncMrBuffers {
  rmm::mr::cuda_async_memory_resource mr{};
  std::unique_ptr<rmm::device_buffer> send;
  std::unique_ptr<rmm::device_buffer> recv;

  static std::shared_ptr<RmmCudaAsyncMrBuffers> allocate(std::size_t messageSize)
  {
    auto p   = std::make_shared<RmmCudaAsyncMrBuffers>();
    p->send  = std::make_unique<rmm::device_buffer>(messageSize, cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p->mr);
    p->recv  = std::make_unique<rmm::device_buffer>(messageSize, cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p->mr);
    initialize_send_pattern(*p->send, messageSize);
    return p;
  }
};

struct RmmCudaAsyncManagedMrBuffers {
  rmm::mr::cuda_async_managed_memory_resource mr{};
  std::unique_ptr<rmm::device_buffer> send;
  std::unique_ptr<rmm::device_buffer> recv;

  static std::shared_ptr<RmmCudaAsyncManagedMrBuffers> allocate(std::size_t messageSize)
  {
    auto p   = std::make_shared<RmmCudaAsyncManagedMrBuffers>();
    p->send  = std::make_unique<rmm::device_buffer>(messageSize, cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p->mr);
    p->recv  = std::make_unique<rmm::device_buffer>(messageSize, cuda::stream_ref{cudaStream_t{cudaStreamDefault}}, p->mr);
    initialize_send_pattern(*p->send, messageSize);
    return p;
  }
};

}  // namespace rmm_benchmark_detail

struct RmmBufferInterfaceBase : public BufferInterface {
  virtual ~RmmBufferInterfaceBase() = default;

  virtual std::unique_ptr<RmmBufferInterfaceBase> clone() const = 0;

  static std::unique_ptr<RmmBufferInterfaceBase> createBufferInterface(MemoryType memoryType,
                                                                     std::size_t messageSize,
                                                                     bool reuseAlloc);

  static std::unique_ptr<RmmBufferInterfaceBase> allocateBuffers(MemoryType memoryType,
                                                                 std::size_t messageSize);
};

template <typename BufferPack>
struct RmmBufferInterface : public RmmBufferInterfaceBase {
  std::shared_ptr<BufferPack> pack;

  explicit RmmBufferInterface(std::shared_ptr<BufferPack> p) : pack(std::move(p)) {}

  void* getSendPtr() override { return pack->send->data(); }

  void* getRecvPtr() override { return pack->recv->data(); }

  void verifyResults(std::size_t messageSize) override
  {
    rmm_benchmark_detail::verify_device_buffers(*pack->send, *pack->recv, messageSize);
  }

  std::unique_ptr<RmmBufferInterfaceBase> clone() const override
  {
    return std::make_unique<RmmBufferInterface<BufferPack>>(pack);
  }
};

inline std::unique_ptr<RmmBufferInterfaceBase> RmmBufferInterfaceBase::allocateBuffers(
  MemoryType memoryType, std::size_t messageSize)
{
  using namespace rmm_benchmark_detail;
  switch (memoryType) {
    case MemoryType::RmmDeviceMemoryResource:
      return std::make_unique<RmmBufferInterface<RmmDeviceMrBuffers>>(
        RmmDeviceMrBuffers::allocate(messageSize));
    case MemoryType::RmmPoolMemoryResource:
      return std::make_unique<RmmBufferInterface<RmmPoolBuffers>>(RmmPoolBuffers::allocate(messageSize));
    case MemoryType::RmmCudaAsyncMemoryResource:
      return std::make_unique<RmmBufferInterface<RmmCudaAsyncMrBuffers>>(
        RmmCudaAsyncMrBuffers::allocate(messageSize));
    case MemoryType::RmmCudaAsyncManagedMemoryResource:
      return std::make_unique<RmmBufferInterface<RmmCudaAsyncManagedMrBuffers>>(
        RmmCudaAsyncManagedMrBuffers::allocate(messageSize));
    default: throw std::runtime_error("Unsupported RMM memory type");
  }
}

inline std::unique_ptr<RmmBufferInterfaceBase> RmmBufferInterfaceBase::createBufferInterface(
  MemoryType memoryType, std::size_t messageSize, bool reuseAlloc)
{
  if (reuseAlloc) {
    static std::unordered_map<MemoryType, std::unique_ptr<RmmBufferInterfaceBase>> reuseBuffers;

    auto it = reuseBuffers.find(memoryType);
    if (it == reuseBuffers.end()) { reuseBuffers[memoryType] = allocateBuffers(memoryType, messageSize); }

    return reuseBuffers[memoryType]->clone();
  }
  return allocateBuffers(memoryType, messageSize);
}

#endif  // UCXX_BENCHMARKS_ENABLE_RMM

#ifdef UCXX_BENCHMARKS_ENABLE_CCCL

namespace cccl_benchmark_detail {

// CCCL benchmarks use nullptr (the default CUDA stream) for async operations because
// UCX is not yet stream-aware (see https://github.com/rapidsai/ucxx/issues/112).
inline void initialize_send_pattern(ucxx::CCCLBuffer& send, std::size_t messageSize)
{
  std::vector<char> pattern(messageSize, 0xaa);
  CUDA_EXIT_ON_ERROR(cudaMemcpyAsync(send.data(),
                                     pattern.data(),
                                     messageSize,
                                     cudaMemcpyHostToDevice,
                                     nullptr),
                     "CCCL send buffer initialization");
  CUDA_EXIT_ON_ERROR(cudaStreamSynchronize(nullptr), "CCCL stream synchronization");
}

inline void verify_device_buffers(ucxx::CCCLBuffer& send,
                                  ucxx::CCCLBuffer& recv,
                                  std::size_t messageSize)
{
  std::vector<char> sendData(messageSize);
  std::vector<char> recvData(messageSize);
  CUDA_EXIT_ON_ERROR(cudaMemcpyAsync(sendData.data(),
                                     send.data(),
                                     messageSize,
                                     cudaMemcpyDeviceToHost,
                                     nullptr),
                     "CCCL send data copy for verification");
  CUDA_EXIT_ON_ERROR(cudaMemcpyAsync(recvData.data(),
                                     recv.data(),
                                     messageSize,
                                     cudaMemcpyDeviceToHost,
                                     nullptr),
                     "CCCL recv data copy for verification");
  CUDA_EXIT_ON_ERROR(cudaStreamSynchronize(nullptr),
                     "CCCL stream synchronization for verification");
  for (std::size_t j = 0; j < messageSize; ++j)
    if (recvData[j] != sendData[j])
      throw std::runtime_error("CCCL data verification failed at byte " + std::to_string(j));
}

struct CcclDevicePoolBuffers {
  std::shared_ptr<ucxx::CCCLBuffer> send;
  std::shared_ptr<ucxx::CCCLBuffer> recv;

  static std::shared_ptr<CcclDevicePoolBuffers> allocate(std::size_t messageSize)
  {
    auto p   = std::make_shared<CcclDevicePoolBuffers>();
    p->send  = std::make_shared<ucxx::CCCLBuffer>(messageSize);
    p->recv  = std::make_shared<ucxx::CCCLBuffer>(messageSize);
    initialize_send_pattern(*p->send, messageSize);
    return p;
  }
};

struct CcclSharedResourceBuffers {
  std::shared_ptr<ucxx::CCCLBuffer> send;
  std::shared_ptr<ucxx::CCCLBuffer> recv;

  static std::shared_ptr<CcclSharedResourceBuffers> allocate(std::size_t messageSize)
  {
    auto p   = std::make_shared<CcclSharedResourceBuffers>();
    p->send  = std::make_shared<ucxx::CCCLBuffer>(messageSize);
    p->recv  = std::make_shared<ucxx::CCCLBuffer>(messageSize);
    initialize_send_pattern(*p->send, messageSize);
    return p;
  }
};

struct CcclCudaAsyncBuffers {
  std::shared_ptr<ucxx::CCCLBuffer> send;
  std::shared_ptr<ucxx::CCCLBuffer> recv;

  static std::shared_ptr<CcclCudaAsyncBuffers> allocate(std::size_t messageSize)
  {
    auto p   = std::make_shared<CcclCudaAsyncBuffers>();
    p->send  = std::make_shared<ucxx::CCCLBuffer>(messageSize);
    p->recv  = std::make_shared<ucxx::CCCLBuffer>(messageSize);
    initialize_send_pattern(*p->send, messageSize);
    return p;
  }
};

struct CcclCudaAsyncManagedBuffers {
  std::shared_ptr<ucxx::CCCLBuffer> send;
  std::shared_ptr<ucxx::CCCLBuffer> recv;

  static std::shared_ptr<CcclCudaAsyncManagedBuffers> allocate(std::size_t messageSize)
  {
    auto p   = std::make_shared<CcclCudaAsyncManagedBuffers>();
    p->send  = std::make_shared<ucxx::CCCLBuffer>(messageSize);
    p->recv  = std::make_shared<ucxx::CCCLBuffer>(messageSize);
    initialize_send_pattern(*p->send, messageSize);
    return p;
  }
};

}  // namespace cccl_benchmark_detail

struct CcclBufferInterfaceBase : public BufferInterface {
  virtual ~CcclBufferInterfaceBase() = default;

  virtual std::unique_ptr<CcclBufferInterfaceBase> clone() const = 0;

  static std::unique_ptr<CcclBufferInterfaceBase> createBufferInterface(MemoryType memoryType,
                                                                        std::size_t messageSize,
                                                                        bool reuseAlloc);

  static std::unique_ptr<CcclBufferInterfaceBase> allocateBuffers(MemoryType memoryType,
                                                                  std::size_t messageSize);
};

template <typename BufferPack>
struct CcclBufferInterface : public CcclBufferInterfaceBase {
  std::shared_ptr<BufferPack> pack;

  explicit CcclBufferInterface(std::shared_ptr<BufferPack> p) : pack(std::move(p)) {}

  void* getSendPtr() override { return pack->send->data(); }

  void* getRecvPtr() override { return pack->recv->data(); }

  void verifyResults(std::size_t messageSize) override
  {
    cccl_benchmark_detail::verify_device_buffers(*pack->send, *pack->recv, messageSize);
  }

  std::unique_ptr<CcclBufferInterfaceBase> clone() const override
  {
    return std::make_unique<CcclBufferInterface<BufferPack>>(pack);
  }
};

inline std::unique_ptr<CcclBufferInterfaceBase> CcclBufferInterfaceBase::allocateBuffers(
  MemoryType memoryType, std::size_t messageSize)
{
  using namespace cccl_benchmark_detail;
  switch (memoryType) {
    case MemoryType::CcclDeviceMemoryPool:
      return std::make_unique<CcclBufferInterface<CcclDevicePoolBuffers>>(
        CcclDevicePoolBuffers::allocate(messageSize));
    case MemoryType::CcclSharedResource:
      return std::make_unique<CcclBufferInterface<CcclSharedResourceBuffers>>(
        CcclSharedResourceBuffers::allocate(messageSize));
    case MemoryType::CcclCudaAsyncResource:
      return std::make_unique<CcclBufferInterface<CcclCudaAsyncBuffers>>(
        CcclCudaAsyncBuffers::allocate(messageSize));
    case MemoryType::CcclCudaAsyncManagedResource:
      return std::make_unique<CcclBufferInterface<CcclCudaAsyncManagedBuffers>>(
        CcclCudaAsyncManagedBuffers::allocate(messageSize));
    default: throw std::runtime_error("Unsupported CCCL memory type");
  }
}

inline std::unique_ptr<CcclBufferInterfaceBase> CcclBufferInterfaceBase::createBufferInterface(
  MemoryType memoryType, std::size_t messageSize, bool reuseAlloc)
{
  if (reuseAlloc) {
    static std::unordered_map<MemoryType, std::unique_ptr<CcclBufferInterfaceBase>> reuseBuffers;

    auto it = reuseBuffers.find(memoryType);
    if (it == reuseBuffers.end()) { reuseBuffers[memoryType] = allocateBuffers(memoryType, messageSize); }

    return reuseBuffers[memoryType]->clone();
  }
  return allocateBuffers(memoryType, messageSize);
}

#endif  // UCXX_BENCHMARKS_ENABLE_CCCL
