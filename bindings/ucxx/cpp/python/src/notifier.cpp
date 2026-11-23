/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <mutex>
#include <utility>

#include <ucxx/log.h>
#include <ucxx/python/notifier.h>
#include <ucxx/python/python_future.h>

namespace ucxx {

namespace python {

std::shared_ptr<::ucxx::Notifier> createNotifier()
{
  return std::shared_ptr<::ucxx::Notifier>(new ::ucxx::python::Notifier());
}

Notifier::~Notifier() {}

void Notifier::scheduleFutureNotify(std::shared_ptr<::ucxx::Future> future, ucs_status_t status)
{
  ucxx_trace_req("ucxx::python::Notifier::%s, future: %p, handle: %p",
                 __func__,
                 future.get(),
                 future->getHandle());
  auto p = std::make_pair(future, status);
  {
    std::lock_guard<std::mutex> lock(_notifierThreadMutex);
    _notifierThreadFutureStatus.push_back(p);
    _notifierThreadFutureStatusReady = true;
  }
  _notifierThreadConditionVariable.notify_one();
  ucxx_trace_req("ucxx::python::Notifier::%s, notified future: %p, handle: %p",
                 __func__,
                 future.get(),
                 future->getHandle());
}

void Notifier::runRequestNotifier()
{
  decltype(_notifierThreadFutureStatus) notifierThreadFutureStatus;
  {
    std::unique_lock<std::mutex> lock(_notifierThreadMutex);
    notifierThreadFutureStatus = std::move(_notifierThreadFutureStatus);
  }

  ucxx_trace_req(
    "ucxx::python::Notifier::%s, notifying %lu", __func__, notifierThreadFutureStatus.size());
  for (auto& p : notifierThreadFutureStatus) {
    // r->future_set_result;
    p.first->set(p.second);
    ucxx_trace_req("ucxx::python::Notifier::%s, notified future: %p, handle: %p",
                   __func__,
                   p.first.get(),
                   p.first->getHandle());
  }
}

RequestNotifierWaitState Notifier::waitRequestNotifierWithoutTimeout()
{
  ucxx_trace_req("ucxx::python::Notifier::%s", __func__);

  std::unique_lock<std::mutex> lock(_notifierThreadMutex);
  _notifierThreadConditionVariable.wait(lock, [this] {
    return _notifierThreadFutureStatusReady ||
           _notifierThreadFutureStatusFinished == RequestNotifierThreadState::Stopping;
  });

  auto state = _notifierThreadFutureStatusReady ? RequestNotifierWaitState::Ready
                                                : RequestNotifierWaitState::Shutdown;

  ucxx_trace_req("ucxx::python::Notifier::%s, unlock: %d", __func__, static_cast<int>(state));
  _notifierThreadFutureStatusReady = false;

  return state;
}

RequestNotifierWaitState Notifier::waitRequestNotifierWithTimeout(uint64_t period)
{
  ucxx_trace_req("ucxx::python::Notifier::%s", __func__);

  std::unique_lock<std::mutex> lock(_notifierThreadMutex);
  bool condition = _notifierThreadConditionVariable.wait_for(
    lock, std::chrono::duration<uint64_t, std::nano>(period), [this] {
      return _notifierThreadFutureStatusReady ||
             _notifierThreadFutureStatusFinished == RequestNotifierThreadState::Stopping;
    });

  auto state = (condition ? (_notifierThreadFutureStatusReady ? RequestNotifierWaitState::Ready
                                                              : RequestNotifierWaitState::Shutdown)
                          : RequestNotifierWaitState::Timeout);

  ucxx_trace_req("ucxx::python::Notifier::%s, unlock: %d", __func__, static_cast<int>(state));
  if (state == RequestNotifierWaitState::Ready) _notifierThreadFutureStatusReady = false;

  return state;
}

RequestNotifierWaitState Notifier::waitRequestNotifier(uint64_t period)
{
  ucxx_trace_req("ucxx::python::Notifier::%s", __func__);

  if (_notifierThreadFutureStatusFinished == RequestNotifierThreadState::Stopping) {
    _notifierThreadFutureStatusFinished = RequestNotifierThreadState::Running;
    return RequestNotifierWaitState::Shutdown;
  }

  return (period > 0) ? waitRequestNotifierWithTimeout(period)
                      : waitRequestNotifierWithoutTimeout();
}

void Notifier::stopRequestNotifierThread()
{
  {
    std::lock_guard<std::mutex> lock(_notifierThreadMutex);
    _notifierThreadFutureStatusFinished = RequestNotifierThreadState::Stopping;
  }
  _notifierThreadConditionVariable.notify_all();
}

bool Notifier::isRunning() const
{
  return _notifierThreadFutureStatusReady ||
         _notifierThreadFutureStatusFinished == RequestNotifierThreadState::Running;
}

}  // namespace python

}  // namespace ucxx
