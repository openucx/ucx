/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdio>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ucp/api/ucp.h>

#include <internal/constructors.h>
#include <ucxx/remote_key.h>
#include <ucxx/utils/ucx.h>

namespace ucxx {

RemoteKey::RemoteKey(std::shared_ptr<MemoryHandle> memoryHandle)
  : _memoryBaseAddress(memoryHandle->getBaseAddress()), _memorySize(memoryHandle->getSize())
{
  setParent(memoryHandle);

  utils::ucsErrorThrow(
    ucp_rkey_pack(std::static_pointer_cast<Context>(memoryHandle->getParent())->getHandle(),
                  memoryHandle->getHandle(),
                  &_packedRemoteKey,
                  &_packedRemoteKeySize));

  ucxx_trace(
    "ucxx::RemoteKey created (memory handle): %p, base address: 0x%lx, size: %lu, packed remote "
    "key "
    "size: %lu",
    this,
    _memoryBaseAddress,
    _memorySize,
    _packedRemoteKeySize);
}

RemoteKey::RemoteKey(std::shared_ptr<Endpoint> endpoint, SerializedRemoteKey serializedRemoteKey)
{
  setParent(endpoint);

  deserialize(serializedRemoteKey);

  utils::ucsErrorThrow(ucp_ep_rkey_unpack(endpoint->getHandle(), _packedRemoteKey, &_remoteKey));

  ucxx_trace(
    "ucxx::RemoteKey created (deserialize): %p, UCP handle: %p, base address: 0x%lx, size: %lu, "
    "packed remote key size: %lu",
    this,
    _remoteKey,
    _memoryBaseAddress,
    _memorySize,
    _packedRemoteKeySize);
}

RemoteKey::~RemoteKey()
{
  // ucxx_trace("ucxx::Endpoint destroyed: %p, UCP handle: %p", this, _originalHandle);
  if (std::dynamic_pointer_cast<MemoryHandle>(getParent()) != nullptr) {
    // Only packed remote key if this object was created from a `MemoryHandle`, i.e., the
    // buffer is local.
    ucp_rkey_buffer_release(_packedRemoteKey);
    ucxx_trace("ucxx::RemoteKey destroyed (memory handle): %p", this);
  }
  if (_remoteKey != nullptr) {
    // Only destroy remote key if this was created from a `SerializedRemoteKey`, i.e., the
    // buffer is remote.
    ucp_rkey_destroy(_remoteKey);
    ucxx_trace("ucxx::RemoteKey destroyed (deserialized): %p, UCP handle: %p", this, _remoteKey);
  }
}

namespace detail {

std::shared_ptr<RemoteKey> ConstructorFactory::createRemoteKeyFromMemoryHandle(
  std::shared_ptr<MemoryHandle> memoryHandle)
{
  return std::shared_ptr<RemoteKey>(new RemoteKey(memoryHandle));
}

std::shared_ptr<RemoteKey> ConstructorFactory::createRemoteKeyFromSerialized(
  std::shared_ptr<Endpoint> endpoint, SerializedRemoteKey serializedRemoteKey)
{
  return std::shared_ptr<RemoteKey>(new RemoteKey(endpoint, serializedRemoteKey));
}

}  // namespace detail

size_t RemoteKey::getSize() const { return _memorySize; }

uint64_t RemoteKey::getBaseAddress() { return _memoryBaseAddress; }

ucp_rkey_h RemoteKey::getHandle() { return _remoteKey; }

SerializedRemoteKey RemoteKey::serialize() const
{
  std::stringstream ss;

  if (_packedRemoteKeySize > std::numeric_limits<std::streamsize>::max())
    // We should never have a remote key this big, but just in case.
    throw std::overflow_error("Remote key is too large to deserialize");

  ss.write(reinterpret_cast<char const*>(&_packedRemoteKeySize), sizeof(_packedRemoteKeySize));
  ss.write(reinterpret_cast<char const*>(_packedRemoteKey),
           static_cast<std::streamsize>(_packedRemoteKeySize));
  ss.write(reinterpret_cast<char const*>(&_memoryBaseAddress), sizeof(_memoryBaseAddress));
  ss.write(reinterpret_cast<char const*>(&_memorySize), sizeof(_memorySize));

  auto serializedString = ss.str();

  // Hash data to provide some degree of confidence on received data.
  std::stringstream ssHash;
  std::hash<std::string> hasher;
  SerializedRemoteKeyHash hash = hasher(serializedString);
  ssHash.write(reinterpret_cast<char const*>(&hash), sizeof(hash));

  return ssHash.str() + serializedString;
}

void RemoteKey::deserialize(const SerializedRemoteKey& serializedRemoteKey)
{
  auto serializedRemoteKeyHash = std::string(
    serializedRemoteKey.begin(), serializedRemoteKey.begin() + sizeof(SerializedRemoteKeyHash));
  auto serializedRemoteKeyData = std::string(
    serializedRemoteKey.begin() + sizeof(SerializedRemoteKeyHash), serializedRemoteKey.end());

  // Check data hash and throw if there's no match.
  std::stringstream ss{serializedRemoteKeyHash};
  SerializedRemoteKeyHash expectedHash;
  ss.read(reinterpret_cast<char*>(&expectedHash), sizeof(expectedHash));
  std::hash<std::string> hasher;
  SerializedRemoteKeyHash actualHash = hasher(serializedRemoteKeyData);
  if (actualHash != expectedHash)
    throw std::runtime_error("Checksum error of serialized remote key");

  ss = std::stringstream{std::string(serializedRemoteKey.begin() + sizeof(SerializedRemoteKeyHash),
                                     serializedRemoteKey.end())};

  ss.read(reinterpret_cast<char*>(&_packedRemoteKeySize), sizeof(_packedRemoteKeySize));

  const size_t fixedFieldsSize =
    sizeof(_packedRemoteKeySize) + sizeof(_memoryBaseAddress) + sizeof(_memorySize);
  if (serializedRemoteKeyData.size() < fixedFieldsSize ||
      _packedRemoteKeySize > serializedRemoteKeyData.size() - fixedFieldsSize)
    throw std::overflow_error("Packed remote key size exceeds serialized data length");

  _packedRemoteKeyVector = std::vector<char>(_packedRemoteKeySize);
  _packedRemoteKey       = _packedRemoteKeyVector.data();

  ss.read(reinterpret_cast<char*>(_packedRemoteKey),
          static_cast<std::streamsize>(_packedRemoteKeySize));
  ss.read(reinterpret_cast<char*>(&_memoryBaseAddress), sizeof(_memoryBaseAddress));
  ss.read(reinterpret_cast<char*>(&_memorySize), sizeof(_memorySize));
}

}  // namespace ucxx
