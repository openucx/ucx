/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <cstdio>
#include <string>

namespace ucxx {

namespace utils {

/**
 * @brief Create a file descriptor from a temporary file.
 *
 * Create a file descriptor from a temporary file.
 *
 * @throws std::ios_base::failure if creating a temporary file fails, a common cause being
 *                                lack of write permissions to `/tmp`.
 *
 * @returns The file descriptor created.
 */
[[nodiscard]] FILE* createTextFileDescriptor();

/**
 * @brief Decode text file descriptor.
 *
 * Decode a text file descriptor and return it as a string.
 *
 * @throws std::ios_base::failure if reading the file descriptor fails.
 *
 * @returns The string with a copy of the file descriptor contents.
 */
[[nodiscard]] std::string decodeTextFileDescriptor(FILE* textFileDescriptor);

}  // namespace utils

}  // namespace ucxx
