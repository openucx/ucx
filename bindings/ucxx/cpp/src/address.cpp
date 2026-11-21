/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <internal/constructors.h>
#include <ucxx/address.h>
#include <ucxx/utils/ucx.h>

namespace ucxx {

Address::Address(std::shared_ptr<Worker> worker, ucp_address_t* address, size_t length)
  : _handle{address}, _length{length}
{
  if (worker != nullptr) setParent(worker);
}

Address::~Address()
{
  if (_handle == nullptr) return;

  auto worker = std::dynamic_pointer_cast<Worker>(getParent());
  if (worker == nullptr) {
    delete[] reinterpret_cast<char*>(_handle);
  } else {
    ucp_worker_release_address(worker->getHandle(), _handle);
  }
}

namespace detail {

std::shared_ptr<Address> ConstructorFactory::createAddressFromWorker(std::shared_ptr<Worker> worker)
{
  ucp_worker_h ucp_worker = worker->getHandle();
  ucp_address_t* address{nullptr};
  size_t length = 0;

  utils::ucsErrorThrow(ucp_worker_get_address(ucp_worker, &address, &length));
  return std::shared_ptr<Address>(new Address(worker, address, length));
}

std::shared_ptr<Address> ConstructorFactory::createAddressFromString(std::string_view addressString)
{
  size_t length = addressString.length();
  if (length == 0) throw std::invalid_argument("UCP address must not be empty");

  ucp_address_t* address = reinterpret_cast<ucp_address_t*>(new char[length]);
  memcpy(address, addressString.data(), length);
  return std::shared_ptr<Address>(new Address(nullptr, address, length));
}

}  // namespace detail

ucp_address_t* Address::getHandle() const { return _handle; }

size_t Address::getLength() const { return _length; }

std::string_view Address::getStringView() const
{
  return std::string_view{reinterpret_cast<const char*>(_handle), _length};
}

}  // namespace ucxx
