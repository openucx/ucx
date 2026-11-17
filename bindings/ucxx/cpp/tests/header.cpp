/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <algorithm>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ucxx/api.h>

using ::testing::ContainerEq;

namespace {

TEST(HeaderTest, DataSize)
{
  const bool next         = false;
  const size_t framesSize = 1;
  std::vector<int> isCUDA{0};
  std::vector<size_t> size{1};

  const ucxx::Header header(next, framesSize, isCUDA.data(), size.data());

  const size_t ExpectedDataSize =
    sizeof(header.next) + sizeof(header.nframes) + (sizeof(header.isCUDA) + sizeof(header.size));

  ASSERT_EQ(header.dataSize(), ExpectedDataSize);
}

TEST(HeaderTest, PointerConstructor)
{
  const bool next         = false;
  const size_t framesSize = 5;
  std::vector<int> isCUDA{1, 0, 1, 0, 1};
  std::vector<size_t> size{1, 2, 3, 4, 5};

  const ucxx::Header header(next, framesSize, isCUDA.data(), size.data());

  std::vector<int> headerIsCUDA(header.isCUDA.begin(), header.isCUDA.begin() + framesSize);
  std::vector<size_t> headerSize(header.size.begin(), header.size.begin() + framesSize);

  ASSERT_EQ(header.next, next);
  ASSERT_EQ(header.nframes, framesSize);
  ASSERT_THAT(headerIsCUDA, ContainerEq(isCUDA));
  ASSERT_THAT(headerSize, ContainerEq(size));

  auto serialized   = header.serialize();
  auto deserialized = ucxx::Header(serialized);

  ASSERT_EQ(deserialized.dataSize(), header.dataSize());
  ASSERT_EQ(deserialized.next, header.next);
  ASSERT_EQ(deserialized.nframes, header.nframes);
  ASSERT_THAT(deserialized.isCUDA, ContainerEq(header.isCUDA));
  ASSERT_THAT(deserialized.size, ContainerEq(header.size));
}

class FromPointerGenerator : public ::testing::Test, public ::testing::WithParamInterface<size_t> {
 private:
  void generateData()
  {
    _isCUDA.resize(_framesSize);
    _size.resize(_framesSize);

    std::iota(_size.begin(), _size.end(), 0);
    std::generate(_isCUDA.begin(), _isCUDA.end(), [n = 0]() mutable { return n++ % 2; });

    _headers = std::move(ucxx::Header::buildHeaders(_size, _isCUDA));
  }

 protected:
  size_t _framesSize;
  std::vector<size_t> _size;
  std::vector<int> _isCUDA;
  std::vector<ucxx::Header> _headers;

  void SetUp()
  {
    _framesSize = GetParam();
    generateData();
  }
};

TEST_P(FromPointerGenerator, PointerConstructor)
{
  for (size_t i = 0; i < _headers.size(); ++i) {
    const auto& header = _headers[i];

    const bool next = i != _headers.size() - 1;
    const size_t expectedNumFrames =
      header.next ? ucxx::HeaderFramesSize : _framesSize - i * ucxx::HeaderFramesSize;
    const size_t firstIdx = i * ucxx::HeaderFramesSize;
    const size_t lastIdx  = std::min((i + 1) * ucxx::HeaderFramesSize, _framesSize);

    auto serialized   = header.serialize();
    auto deserialized = ucxx::Header(serialized);

    // Assert next
    ASSERT_EQ(header.next, next);
    ASSERT_EQ(deserialized.next, header.next);

    // Assert number of frames
    ASSERT_EQ(header.nframes, expectedNumFrames);
    ASSERT_EQ(deserialized.nframes, header.nframes);

    // Assert isCUDA
    std::vector<int> expectedIsCUDA(std::cbegin(_isCUDA) + firstIdx,
                                    std::cbegin(_isCUDA) + lastIdx);
    std::vector<int> headerIsCUDA(header.isCUDA.begin(), header.isCUDA.begin() + expectedNumFrames);
    ASSERT_THAT(headerIsCUDA, ContainerEq(expectedIsCUDA));
    std::vector<int> deserializedIsCUDA(deserialized.isCUDA.begin(),
                                        deserialized.isCUDA.begin() + expectedNumFrames);
    ASSERT_THAT(deserializedIsCUDA, ContainerEq(headerIsCUDA));

    // Assert size
    std::vector<size_t> expectedSize(std::cbegin(_size) + firstIdx, std::cbegin(_size) + lastIdx);
    std::vector<size_t> headerSize(header.size.begin(), header.size.begin() + expectedNumFrames);
    ASSERT_THAT(headerSize, ContainerEq(expectedSize));
    std::vector<size_t> deserializedSize(deserialized.size.begin(),
                                         deserialized.size.begin() + expectedNumFrames);
    ASSERT_THAT(deserializedSize, ContainerEq(headerSize));
  }
}

INSTANTIATE_TEST_SUITE_P(SingleFrame,
                         FromPointerGenerator,
                         testing::Values(0, 1, 5, 10, 100, 101, 200, 201));

TEST(HeaderTest, DeserializeOversizedNframes)
{
  std::stringstream ss;
  bool next      = false;
  size_t nframes = ucxx::HeaderFramesSize + 1;
  ss.write(reinterpret_cast<const char*>(&next), sizeof(next));
  ss.write(reinterpret_cast<const char*>(&nframes), sizeof(nframes));
  EXPECT_THROW(ucxx::Header(ss.str()), std::overflow_error);
}

}  // namespace
