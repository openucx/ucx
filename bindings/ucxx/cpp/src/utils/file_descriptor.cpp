/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdio>
#include <exception>
#include <ios>
#include <string>

#include <ucxx/utils/file_descriptor.h>

namespace ucxx {

namespace utils {

FILE* createTextFileDescriptor()
{
  FILE* textFileDescriptor = std::tmpfile();
  if (textFileDescriptor == nullptr) throw std::ios_base::failure("tmpfile() failed");

  return textFileDescriptor;
}

std::string decodeTextFileDescriptor(FILE* textFileDescriptor)
{
  rewind(textFileDescriptor);
  fseek(textFileDescriptor, 0, SEEK_END);
  int64_t pos = ftell(textFileDescriptor);
  if (pos == -1L) throw std::ios_base::failure("ftell() failed");
  size_t size = static_cast<size_t>(pos);
  rewind(textFileDescriptor);

  std::string textString(size, '\0');

  if (fread(&textString[0], sizeof(char), size, textFileDescriptor) != size)
    throw std::ios_base::failure("fread() failed");

  fclose(textFileDescriptor);

  return textString;
}

}  // namespace utils

}  // namespace ucxx
