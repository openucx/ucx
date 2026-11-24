/**
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>

namespace ucxx {

namespace utils {

/*
 * @brief Check whether Python support is available.
 *
 * Check that binary was built with Python support and `libucxx_python.so` is in the
 * library path. The check is done by attempting to `dlopen` the library, returning whether
 * both conditions are met.
 *
 * @returns whether Python support is available.
 */
[[nodiscard]] bool isPythonAvailable();

}  // namespace utils

}  // namespace ucxx
