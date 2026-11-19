/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <vector>

#include <ucxx/request.h>

namespace ucxx {

void waitSingleRequest(std::shared_ptr<Worker> worker, std::shared_ptr<Request> request)
{
  while (!request->isCompleted())
    worker->progress();
  // while (!request->isCompleted());

  request->checkError();
}

void waitRequests(std::shared_ptr<Worker> worker, std::vector<std::shared_ptr<Request>> requests)
{
  for (auto& r : requests)
    waitSingleRequest(worker, r);
}

}  // namespace ucxx
