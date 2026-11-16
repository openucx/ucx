/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <algorithm>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>

#include <gtest/gtest.h>

#include <ucxx/api.h>

namespace {

class BufferAllocator : public ::testing::Test,
                        public ::testing::WithParamInterface<std::pair<ucxx::BufferType, size_t>> {
 protected:
  ucxx::BufferType _type;
  size_t _size;
  std::shared_ptr<ucxx::Buffer> _buffer;

  void SetUp()
  {
    auto param = GetParam();
    _type      = param.first;
    _size      = param.second;

    _buffer = allocateBuffer(_type, _size);
  }
};

TEST_P(BufferAllocator, TestType)
{
  ASSERT_EQ(_buffer->getType(), _type);

  if (_type == ucxx::BufferType::Host) {
    auto buffer = std::dynamic_pointer_cast<ucxx::HostBuffer>(_buffer);
    ASSERT_EQ(buffer->getType(), _type);

    auto releasedBuffer = buffer->release();

    ASSERT_EQ(buffer->getType(), ucxx::BufferType::Invalid);

    free(releasedBuffer);
  } else if (_type == ucxx::BufferType::CCCL) {
#if UCXX_ENABLE_CCCL
    auto buffer = std::dynamic_pointer_cast<ucxx::CCCLBuffer>(_buffer);
    ASSERT_EQ(buffer->getType(), _type);
    return;  // CCCLBuffer uses PIMPL - lifetime managed by unique_ptr, no release() needed
#else
    GTEST_SKIP() << "UCXX was not built with CCCL support";
#endif
  }

  ASSERT_EQ(_buffer->getType(), ucxx::BufferType::Invalid);
}

TEST_P(BufferAllocator, TestSize)
{
  ASSERT_EQ(_buffer->getSize(), _size);

  if (_type == ucxx::BufferType::Host) {
    auto buffer = std::dynamic_pointer_cast<ucxx::HostBuffer>(_buffer);
    ASSERT_EQ(buffer->getSize(), _size);

    auto releasedBuffer = buffer->release();

    ASSERT_EQ(buffer->getSize(), 0u);

    free(releasedBuffer);
  } else if (_type == ucxx::BufferType::CCCL) {
#if UCXX_ENABLE_CCCL
    auto buffer = std::dynamic_pointer_cast<ucxx::CCCLBuffer>(_buffer);
    ASSERT_EQ(buffer->getSize(), _size);
    return;  // CCCLBuffer does not expose release(); post-release assertions do not apply
#else
    GTEST_SKIP() << "UCXX was not built with CCCL support";
#endif
  }

  ASSERT_EQ(_buffer->getSize(), 0u);
}

TEST_P(BufferAllocator, TestData)
{
  ASSERT_NE(_buffer->data(), nullptr);

  if (_type == ucxx::BufferType::Host) {
    auto buffer = std::dynamic_pointer_cast<ucxx::HostBuffer>(_buffer);
    ASSERT_EQ(buffer->data(), _buffer->data());

    auto releasedBuffer = buffer->release();

    ASSERT_NE(releasedBuffer, nullptr);

    free(releasedBuffer);
  } else if (_type == ucxx::BufferType::CCCL) {
#if UCXX_ENABLE_CCCL
    auto buffer = std::dynamic_pointer_cast<ucxx::CCCLBuffer>(_buffer);
    ASSERT_EQ(buffer->data(), _buffer->data());
    return;  // CCCLBuffer does not expose release(); post-release assertions do not apply
#else
    GTEST_SKIP() << "UCXX was not built with CCCL support";
#endif
  }

  EXPECT_THROW(_buffer->data(), std::runtime_error);
}

TEST_P(BufferAllocator, TestThrowAfterRelease)
{
  if (_type == ucxx::BufferType::Host) {
    auto buffer         = std::dynamic_pointer_cast<ucxx::HostBuffer>(_buffer);
    auto releasedBuffer = buffer->release();

    EXPECT_THROW(buffer->data(), std::runtime_error);
    EXPECT_THROW(std::ignore = buffer->release(), std::runtime_error);

    free(releasedBuffer);
  } else if (_type == ucxx::BufferType::CCCL) {
#if UCXX_ENABLE_CCCL
    GTEST_SKIP() << "CCCLBuffer does not expose release()";
#else
    GTEST_SKIP() << "UCXX was not built with CCCL support";
#endif
  }
  EXPECT_THROW(_buffer->data(), std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(Host,
                         BufferAllocator,
                         testing::Values(std::make_pair(ucxx::BufferType::Host, 1),
                                         std::make_pair(ucxx::BufferType::Host, 1000),
                                         std::make_pair(ucxx::BufferType::Host, 1000000)));

#if UCXX_ENABLE_CCCL
INSTANTIATE_TEST_SUITE_P(CCCL,
                         BufferAllocator,
                         testing::Values(std::make_pair(ucxx::BufferType::CCCL, 1),
                                         std::make_pair(ucxx::BufferType::CCCL, 1000),
                                         std::make_pair(ucxx::BufferType::CCCL, 1000000)));
#endif

}  // namespace
