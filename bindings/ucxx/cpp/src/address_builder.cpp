/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <internal/constructors.h>
#include <ucxx/address_builder.h>
#include <ucxx/detail/builder_utils.h>

namespace ucxx {

enum class AddressBuilderSource { Worker, String };

struct AddressBuilder::Impl {
  AddressBuilderSource source;
  std::shared_ptr<Worker> worker{nullptr};
  std::string addressString{};

  explicit Impl(std::shared_ptr<Worker> w)
    : source(AddressBuilderSource::Worker), worker(std::move(w))
  {
  }

  explicit Impl(std::string_view address)
    : source(AddressBuilderSource::String), addressString(address)
  {
  }
};

AddressBuilder::AddressBuilder(std::shared_ptr<Worker> worker)
  : _impl(std::make_unique<Impl>(std::move(worker)))
{
}

AddressBuilder::AddressBuilder(std::string_view addressString)
  : _impl(std::make_unique<Impl>(addressString))
{
}

UCXX_BUILDER_PIMPL_DEFAULTS(AddressBuilder, Address)

std::shared_ptr<Address> AddressBuilder::build()
{
  switch (_impl->source) {
    case AddressBuilderSource::Worker:
      return detail::ConstructorFactory::createAddressFromWorker(_impl->worker);
    case AddressBuilderSource::String:
      return detail::ConstructorFactory::createAddressFromString(_impl->addressString);
  }

  throw std::logic_error("Invalid AddressBuilder source");
}

}  // namespace ucxx
