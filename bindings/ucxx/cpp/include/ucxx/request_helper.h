/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <vector>

#include <ucxx/request.h>
#include <ucxx/worker.h>

namespace ucxx {

/**
 * @brief Wait for a single request to complete.
 *
 * Block while waiting for a single request to complete.
 *
 * @throws ucxx::Error  a specific error if the request failed.
 *
 * @param[in] worker  the worker to progress until completion.
 * @param[in] request the request to wait for.
 */
void waitSingleRequest(std::shared_ptr<Worker> worker, std::shared_ptr<Request> request);

/**
 * @brief Wait for a multiple requests to complete.
 *
 * Block while waiting for all requests to complete.
 *
 * @throws ucxx::Error  the specific error of the first request that failed.
 *
 * @param[in] worker    the worker to progress until completion.
 * @param[in] requests  the requests to wait for.
 */
void waitRequests(std::shared_ptr<Worker> worker, std::vector<std::shared_ptr<Request>> requests);

}  // namespace ucxx
