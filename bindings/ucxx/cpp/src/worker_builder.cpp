/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>

#include <internal/constructors.h>
#include <ucxx/detail/builder_utils.h>
#include <ucxx/worker.h>
#include <ucxx/worker_builder.h>

namespace ucxx {

struct WorkerBuilder::Impl {
  std::shared_ptr<Context> context;
  bool enableDelayedSubmission{false};
  bool enableFuture{false};
  bool enableRequestAttributes{false};
  BufferType cudaBufferType{BufferType::Invalid};

  explicit Impl(std::shared_ptr<Context> ctx) : context(std::move(ctx)) {}
};

WorkerBuilder::WorkerBuilder(std::shared_ptr<Context> context)
  : _impl(std::make_unique<Impl>(std::move(context)))
{
}

UCXX_BUILDER_PIMPL_DEFAULTS(WorkerBuilder, Worker)

WorkerBuilder& WorkerBuilder::delayedSubmission(bool enable)
{
  _impl->enableDelayedSubmission = enable;
  return *this;
}

WorkerBuilder& WorkerBuilder::pythonFuture(bool enable)
{
  _impl->enableFuture = enable;
  return *this;
}

WorkerBuilder& WorkerBuilder::requestAttributes(bool enable)
{
  _impl->enableRequestAttributes = enable;
  return *this;
}

WorkerBuilder& WorkerBuilder::cudaBufferType(BufferType bufferType)
{
  _impl->cudaBufferType = bufferType;
  return *this;
}

std::shared_ptr<Worker> WorkerBuilder::build()
{
  auto worker = detail::ConstructorFactory::createWorker(
    _impl->context, _impl->enableDelayedSubmission, _impl->enableFuture);
  worker->_enableRequestAttributes = _impl->enableRequestAttributes;
  if (_impl->cudaBufferType != BufferType::Invalid)
    worker->setCudaBufferType(_impl->cudaBufferType);
  return worker;
}

}  // namespace ucxx
