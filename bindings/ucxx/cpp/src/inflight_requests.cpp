/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>
#include <vector>

#include <ucxx/inflight_requests.h>
#include <ucxx/log.h>
#include <ucxx/request.h>

namespace ucxx {

InflightRequests::~InflightRequests() { cancelAll(); }

size_t InflightRequests::size()
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _inflight.size();
}

void InflightRequests::insert(const std::shared_ptr<Request>& request)
{
  std::lock_guard<std::mutex> lock(_mutex);
  _inflight.insert(request);
}

void InflightRequests::remove(const std::shared_ptr<Request>& request)
{
  std::lock_guard<std::mutex> lock(_mutex);
  _inflight.erase(request);
}

void InflightRequests::merge(TrackedRequests&& trackedRequests)
{
  std::lock_guard<std::mutex> lock(_mutex);
  for (auto& r : trackedRequests.inflight)
    if (r) _inflight.insert(std::move(r));
  for (auto& r : trackedRequests.canceling)
    if (r) _canceling.insert(std::move(r));
}

size_t InflightRequests::cancelAll()
{
  decltype(_inflight) toCancel;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    toCancel = std::exchange(_inflight, {});
  }

  size_t total = toCancel.size();
  if (total == 0) return 0;

  ucxx_debug("ucxx::InflightRequests::%s, canceling %lu requests", __func__, total);

  for (auto& r : toCancel) {
    if (r) r->cancel();
  }

  {
    std::lock_guard<std::mutex> lock(_mutex);
    for (auto& r : toCancel) {
      if (r && r->getStatus() == UCS_INPROGRESS) _canceling.insert(r);
    }
  }

  return total;
}

TrackedRequests InflightRequests::release()
{
  std::lock_guard<std::mutex> lock(_mutex);
  TrackedRequests result;

  result.inflight.reserve(_inflight.size());
  for (auto& r : _inflight)
    result.inflight.push_back(r);
  _inflight.clear();

  result.canceling.reserve(_canceling.size());
  for (auto& r : _canceling)
    result.canceling.push_back(r);
  _canceling.clear();

  return result;
}

size_t InflightRequests::getCancelingSize()
{
  std::lock_guard<std::mutex> lock(_mutex);

  for (auto it = _canceling.begin(); it != _canceling.end();) {
    if (*it && (*it)->getStatus() != UCS_INPROGRESS)
      it = _canceling.erase(it);
    else
      ++it;
  }

  return _canceling.size();
}

}  // namespace ucxx
