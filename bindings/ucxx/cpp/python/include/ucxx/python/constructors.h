/**
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>

#include <Python.h>

#include <ucxx/buffer.h>

namespace ucxx {

class Context;
class Future;
class Notifier;
class Worker;

namespace python {

[[nodiscard]] std::shared_ptr<::ucxx::Future> createFuture(
  std::shared_ptr<::ucxx::Notifier> notifier);

[[nodiscard]] std::shared_ptr<::ucxx::Future> createFutureWithEventLoop(
  PyObject* asyncioEventLoop, std::shared_ptr<::ucxx::Notifier> notifier);

[[nodiscard]] std::shared_ptr<::ucxx::Notifier> createNotifier();

[[nodiscard]] std::shared_ptr<::ucxx::Worker> createWorker(std::shared_ptr<ucxx::Context> context,
                                                           const bool enableDelayedSubmission,
                                                           const bool enableFuture);

}  // namespace python

}  // namespace ucxx
