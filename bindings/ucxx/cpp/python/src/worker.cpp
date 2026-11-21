/**
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <functional>
#include <ios>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>

#include <Python.h>

#include <ucxx/internal/request_am.h>
#include <ucxx/python/constructors.h>
#include <ucxx/python/future.h>
#include <ucxx/python/worker.h>
#include <ucxx/request_tag.h>
#include <ucxx/utils/python.h>

namespace ucxx {

namespace python {

namespace {

class GilState {
 public:
  GilState() : _state(PyGILState_Ensure()) {}

  GilState(const GilState&)            = delete;
  GilState& operator=(const GilState&) = delete;

  ~GilState() { PyGILState_Release(_state); }

 private:
  PyGILState_STATE _state;
};

}  // namespace

Worker::Worker(std::shared_ptr<Context> context,
               const bool enableDelayedSubmission,
               const bool enableFuture)
  : ::ucxx::Worker(context, enableDelayedSubmission, enableFuture)
{
  if (_enableFuture) _notifier = createNotifier();
}

std::shared_ptr<::ucxx::Worker> createWorker(std::shared_ptr<Context> context,
                                             const bool enableDelayedSubmission,
                                             const bool enableFuture)
{
  auto worker = std::shared_ptr<::ucxx::python::Worker>(
    new ::ucxx::python::Worker(context, enableDelayedSubmission, enableFuture));

  // We can only get a `shared_ptr<Worker>` for the Active Messages callback after it's
  // been created, thus this cannot be in the constructor.
  if (worker->_amData != nullptr) {
    worker->_amData->_worker = worker;

    std::stringstream ownerStream;
    ownerStream << "worker " << worker->getHandle();
    worker->_amData->_ownerString = ownerStream.str();
  }

  return worker;
}

void Worker::populateFuturesPool()
{
  if (!_enableFuture) {
    throw std::runtime_error(
      "Worker future support disabled, please set enableFuture=true when creating the "
      "Worker to use this method.");
  }

  ucxx_trace_req("ucxx::python::Worker::%s, Worker: %p, populateFuturesPool: %p",
                 __func__,
                 this,
                 shared_from_this().get());

  constexpr size_t maxPoolSize = 100;
  constexpr size_t minPoolSize = maxPoolSize / 2;
  size_t futuresNeeded         = 0;
  {
    std::lock_guard<std::mutex> lock(_futuresPoolMutex);
    if (_futuresPool.size() >= minPoolSize) return;
    futuresNeeded = maxPoolSize - _futuresPool.size();
  }

  decltype(_futuresPool) newFuturesPool;
  {
    GilState gil;
    for (size_t i = 0; i < futuresNeeded; ++i)
      newFuturesPool.emplace(createFuture(_notifier));
  }

  std::lock_guard<std::mutex> lock(_futuresPoolMutex);
  while (_futuresPool.size() < maxPoolSize && !newFuturesPool.empty()) {
    _futuresPool.emplace(std::move(newFuturesPool.front()));
    newFuturesPool.pop();
  }
}

void Worker::clearFuturesPool()
{
  if (_enableFuture) {
    ucxx_trace_req("ucxx::python::Worker::%s, Worker: %p, populateFuturesPool: %p",
                   __func__,
                   this,
                   shared_from_this().get());
    decltype(_futuresPool) newFuturesPool;
    {
      std::lock_guard<std::mutex> lock(_futuresPoolMutex);
      std::swap(_futuresPool, newFuturesPool);
    }
  }
}

std::shared_ptr<::ucxx::Future> Worker::getFuture()
{
  if (!_enableFuture) {
    throw std::runtime_error(
      "Worker future support disabled, please set enableFuture=true when creating the "
      "Worker to use this method.");
  }

  auto popFuture = [this]() -> std::shared_ptr<::ucxx::Future> {
    std::lock_guard<std::mutex> lock(_futuresPoolMutex);
    if (_futuresPool.empty()) return nullptr;

    auto ret = _futuresPool.front();
    _futuresPool.pop();
    return ret;
  };

  auto ret = popFuture();
  if (ret == nullptr) {
    ucxx_warn(
      "No Futures available during getFuture(), make sure the Notifier is running "
      "running and calling populateFuturesPool() periodically. Filling futures pool "
      "now, but this may be inefficient.");
    populateFuturesPool();
    ret = popFuture();
  }
  if (ret == nullptr) {
    GilState gil;
    ret = createFuture(_notifier);
  }

  ucxx_trace_req("getFuture: %p %p", ret.get(), ret->getHandle());
  return ret;
}

RequestNotifierWaitState Worker::waitRequestNotifier(uint64_t periodNs)
{
  if (_enableFuture) {
    return _notifier->waitRequestNotifier(periodNs);
  } else {
    throw std::runtime_error(
      "Worker future support disabled, please set enableFuture=true when creating the "
      "Worker to use this method.");
  }
}

void Worker::runRequestNotifier()
{
  if (_enableFuture) {
    _notifier->runRequestNotifier();
  } else {
    throw std::runtime_error(
      "Worker future support disabled, please set enableFuture=true when creating the "
      "Worker to use this method.");
  }
}

void Worker::stopRequestNotifierThread()
{
  if (_enableFuture) {
    _notifier->stopRequestNotifierThread();
  } else {
    throw std::runtime_error(
      "Worker future support disabled, please set enableFuture=true when creating the "
      "Worker to use this method.");
  }
}

}  // namespace python

}  // namespace ucxx
