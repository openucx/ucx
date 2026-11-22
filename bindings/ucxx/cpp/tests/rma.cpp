/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <ucp/api/ucp.h>
#include <ucs/memory/memory_type.h>
#include <ucs/type/status.h>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ucxx/api.h>

#include "include/utils.h"
#include "ucxx/buffer.h"
#include "ucxx/utils/ucx.h"

namespace {

using ::testing::Combine;
using ::testing::Values;

class RmaTest : public ::testing::TestWithParam<std::tuple<ucs_memory_type_t, size_t, bool>> {
 protected:
  std::shared_ptr<ucxx::Context> _context{nullptr};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Endpoint> _ep{nullptr};

  ucs_memory_type_t _memoryType;
  size_t _messageSize;
  bool _preallocateBuffer;
  size_t _rndvThresh{8192};
  void* _buffer{nullptr};

  void SetUp()
  {
    std::tie(_memoryType, _messageSize, _preallocateBuffer) = GetParam();

    _context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags)
                 .configMap({{"RNDV_THRESH", std::to_string(_rndvThresh)}})
                 .build();
    _worker = _context->workerBuilder().build();

    _ep = _worker->endpointBuilder(_worker->addressBuilder().build()).build();

    _buffer = preallocate();
  }

  void TearDown() { release(); }

  void* preallocate()
  {
    if (_preallocateBuffer) {
      if (_memoryType == UCS_MEMORY_TYPE_HOST)
        return malloc(_messageSize);
      else
        throw std::runtime_error("Unsupported memory type");
    }
    return nullptr;
  }

  void release()
  {
    if (_preallocateBuffer && _buffer != nullptr) {
      if (_memoryType == UCS_MEMORY_TYPE_HOST) free(_buffer);
    }
  }
};

class BasicUcxxRmaTest : public ::testing::TestWithParam<std::tuple<size_t>> {
 protected:
  std::shared_ptr<ucxx::Context> _context{nullptr};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Endpoint> _ep{nullptr};

  size_t _messageSize;

  void SetUp()
  {
    std::tie(_messageSize) = GetParam();

    _context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build();
    _worker  = _context->workerBuilder().build();
    _ep      = _worker->endpointBuilder(_worker->addressBuilder().build()).build();
  }
};

TEST_P(RmaTest, MemoryHandle)
{
  auto memoryHandle = _context->memoryHandleBuilder(_messageSize).buffer(_buffer).build();
  ASSERT_GE(memoryHandle->getSize(), _messageSize);
  if (_messageSize == 0)
    ASSERT_EQ(memoryHandle->getBaseAddress(), 0);
  else
    ASSERT_NE(memoryHandle->getBaseAddress(), 0);

  ASSERT_NE(memoryHandle->getHandle(), nullptr);
}

TEST_P(RmaTest, RemoteKey)
{
  auto memoryHandle = _context->memoryHandleBuilder(_messageSize).buffer(_buffer).build();

  auto remoteKey = memoryHandle->remoteKeyBuilder().build();

  ASSERT_EQ(remoteKey->getSize(), memoryHandle->getSize());
  ASSERT_EQ(remoteKey->getBaseAddress(), memoryHandle->getBaseAddress());
  ASSERT_EQ(remoteKey->getHandle(), nullptr);
}

TEST_P(RmaTest, RemoteKeySerialization)
{
  auto memoryHandle = _context->memoryHandleBuilder(_messageSize).buffer(_buffer).build();

  auto remoteKey = memoryHandle->remoteKeyBuilder().build();

  auto serializedRemoteKey = remoteKey->serialize();

  auto deserializedRemoteKey = _ep->remoteKeyBuilder(serializedRemoteKey).build();

  ASSERT_EQ(remoteKey->getSize(), deserializedRemoteKey->getSize());
  ASSERT_EQ(remoteKey->getBaseAddress(), deserializedRemoteKey->getBaseAddress());
  ASSERT_NE(deserializedRemoteKey->getHandle(), nullptr);
}

TEST_P(BasicUcxxRmaTest, RemoteKeyCorruptedSerializedData)
{
  auto memoryHandle = _context->memoryHandleBuilder(_messageSize).buffer(nullptr).build();

  auto remoteKey = memoryHandle->remoteKeyBuilder().build();

  auto serializedRemoteKey = remoteKey->serialize();
  serializedRemoteKey[1]   = ~serializedRemoteKey[1];

  EXPECT_THROW(std::ignore = _ep->remoteKeyBuilder(serializedRemoteKey).build(),
               std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(AttributeTests,
                         RmaTest,
                         Combine(Values(UCS_MEMORY_TYPE_HOST),
                                 Values(0, 1, 4, 4096, 8192, 4194304),
                                 Values(false, true)));

TEST_P(BasicUcxxRmaTest, RemoteKeyTruncatedSerializedData)
{
  // Build a data portion shorter than the fixed fields:
  // packedRemoteKeySize + memoryBaseAddress + memorySize = 24 bytes
  // This ensures no underflow can occur.
  std::stringstream ssData;
  size_t bogusPackedSize = 9999;
  ssData.write(reinterpret_cast<const char*>(&bogusPackedSize), sizeof(bogusPackedSize));
  auto dataStr = ssData.str();

  std::hash<std::string> hasher;
  size_t hash = hasher(dataStr);

  std::stringstream ssFull;
  ssFull.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
  ssFull << dataStr;

  EXPECT_THROW(std::ignore = _ep->remoteKeyBuilder(ssFull.str()).build(), std::overflow_error);
}

INSTANTIATE_TEST_SUITE_P(FailureTests, BasicUcxxRmaTest, Combine(Values(0, 4194304)));

}  // namespace
