/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <vector>

#include "include/utils.h"

void createCudaContextCallback(void* callbackArg)
{
  // Force CUDA context creation
  cudaFree(0);
}

std::function<void()> getProgressFunction(std::shared_ptr<ucxx::Worker> worker,
                                          ProgressMode progressMode)
{
  switch (progressMode) {
    case ProgressMode::Polling: return [worker]() { worker->progress(); };
    case ProgressMode::Blocking: return [worker]() { worker->progressWorkerEvent(-1); };
    case ProgressMode::Wait: return [worker]() { worker->waitProgress(); };
    default: return []() {};
  }
}

bool loopWithTimeout(std::chrono::milliseconds timeout, std::function<bool()> f)
{
  auto startTime = std::chrono::system_clock::now();
  auto endTime   = startTime + std::chrono::milliseconds(timeout);

  while (std::chrono::system_clock::now() < endTime) {
    if (f()) return true;
  }
  return false;
}
