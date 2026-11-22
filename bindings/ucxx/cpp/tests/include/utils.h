/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <functional>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include <ucxx/api.h>

enum class ProgressMode {
  Polling,
  Blocking,
  Wait,
  ThreadPolling,
  ThreadBlocking,
};

void createCudaContextCallback(void* callbackArg);

template <typename RequestType>
inline void waitRequests(std::shared_ptr<ucxx::Worker> worker,
                         const std::vector<std::shared_ptr<RequestType>>& requests,
                         const std::function<void()>& progressWorker)
{
  auto remainingRequests = requests;
  while (!remainingRequests.empty()) {
    auto updatedRequests = std::exchange(remainingRequests, decltype(remainingRequests)());
    for (auto const& r : updatedRequests) {
      if (progressWorker && !r->isCompleted()) progressWorker();
      if (!r->isCompleted())
        remainingRequests.push_back(r);
      else
        r->checkError();
    }
  }
}

std::function<void()> getProgressFunction(std::shared_ptr<ucxx::Worker> worker,
                                          ProgressMode progressMode);

bool loopWithTimeout(std::chrono::milliseconds timeout, std::function<bool()> f);
