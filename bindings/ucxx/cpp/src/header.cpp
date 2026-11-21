/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ucxx/header.h>

namespace ucxx {

Header::Header(bool next, size_t nframes, int* isCUDA, size_t* size) : next{next}, nframes{nframes}
{
  std::copy(isCUDA, isCUDA + nframes, this->isCUDA.begin());
  std::copy(size, size + nframes, this->size.begin());
  if (nframes < HeaderFramesSize) {
    std::fill(this->isCUDA.begin() + nframes, this->isCUDA.begin() + HeaderFramesSize, false);
    std::fill(this->size.begin() + nframes, this->size.begin() + HeaderFramesSize, 0);
  }
}

Header::Header(const std::string& serializedHeader) { deserialize(serializedHeader); }

size_t Header::dataSize() { return sizeof(next) + sizeof(nframes) + sizeof(isCUDA) + sizeof(size); }

const std::string Header::serialize() const
{
  std::stringstream ss;

  ss.write((char const*)&next, sizeof(next));
  ss.write((char const*)&nframes, sizeof(nframes));
  for (size_t i = 0; i < HeaderFramesSize; ++i)
    ss.write((char const*)&isCUDA[i], sizeof(isCUDA[i]));
  for (size_t i = 0; i < HeaderFramesSize; ++i)
    ss.write((char const*)&size[i], sizeof(size[i]));

  return ss.str();
}

void Header::deserialize(const std::string& serializedHeader)
{
  std::stringstream ss{serializedHeader};

  ss.read(reinterpret_cast<char*>(&next), sizeof(next));
  ss.read(reinterpret_cast<char*>(&nframes), sizeof(nframes));
  if (nframes > HeaderFramesSize)
    throw std::overflow_error("Header nframes (" + std::to_string(nframes) +
                              ") exceeds HeaderFramesSize (" + std::to_string(HeaderFramesSize) +
                              ")");
  for (size_t i = 0; i < HeaderFramesSize; ++i)
    ss.read(reinterpret_cast<char*>(&isCUDA[i]), sizeof(isCUDA[i]));
  for (size_t i = 0; i < HeaderFramesSize; ++i)
    ss.read(reinterpret_cast<char*>(&size[i]), sizeof(size[i]));
}

std::vector<Header> Header::buildHeaders(const std::vector<size_t>& size,
                                         const std::vector<int>& isCUDA)
{
  const size_t totalFrames = size.size();

  if (isCUDA.size() != totalFrames)
    throw std::length_error("size and isCUDA must have the same length");

  const size_t totalHeaders = (totalFrames + HeaderFramesSize - 1) / HeaderFramesSize;

  std::vector<Header> headers;

  for (size_t i = 0; i < totalHeaders; ++i) {
    bool hasNext = totalFrames > (i + 1) * HeaderFramesSize;
    size_t headerFrames =
      hasNext ? HeaderFramesSize : HeaderFramesSize - (HeaderFramesSize * (i + 1) - totalFrames);

    size_t idx = i * HeaderFramesSize;
    headers.push_back(Header(hasNext,
                             headerFrames,
                             const_cast<int*>(reinterpret_cast<const int*>(&isCUDA[idx])),
                             const_cast<size_t*>(reinterpret_cast<const size_t*>(&size[idx]))));
  }

  return headers;
}

}  // namespace ucxx
