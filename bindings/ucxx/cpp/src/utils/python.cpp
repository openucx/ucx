/**
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <dlfcn.h>

#include <ucxx/log.h>
#include <ucxx/utils/python.h>

namespace ucxx {

namespace utils {

static bool _ucxxPythonLoadChecked = false;
static void* _ucxxPythonLib        = nullptr;

bool isPythonAvailable()
{
  if (!_ucxxPythonLoadChecked) {
    _ucxxPythonLoadChecked = true;
    _ucxxPythonLib         = dlopen("libucxx_python.so", RTLD_LAZY);
    if (_ucxxPythonLib == nullptr)
      ucxx_debug("dlopen('libucxx_python.so') failed");
    else
      ucxx_debug("dlopen('libucxx_python.so') loaded at %p", _ucxxPythonLib);
  }
  return _ucxxPythonLib != nullptr;
}

}  // namespace utils

}  // namespace ucxx
