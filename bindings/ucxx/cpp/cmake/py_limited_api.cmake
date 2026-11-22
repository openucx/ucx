# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# cmake-format: on
# =================================================================================

# Sets Py_LIMITED_API=<hex> on <target> when SKBUILD_SABI_VERSION is defined, matching the version
# encoding used by Python's C API headers.
#
# This ensures a compile-time error if anything in libpython that isn't part of the Limited API is
# used.
#
# 'scikit-build-core' sets that variable based on 'skbuild.wheel.py-api'.
function(rapids_target_set_py_limited_api target)
  if(NOT DEFINED SKBUILD_SABI_VERSION OR "${SKBUILD_SABI_VERSION}" STREQUAL "")
    message(
      STATUS
        "rapids_target_set_py_limited_api: SKBUILD_SABI_VERSION not set (not using scikit-build-core or 'skbuild.wheel.py-api' not set)"
    )
    return()
  endif()
  message(
    STATUS
      "rapids_target_set_py_limited_api: SKBUILD_SABI_VERSION=${SKBUILD_SABI_VERSION}, setting Py_LIMITED_API on '${target}'"
  )

  # SKBUILD_SABI_VERSION is usually a Python major.minor, e.g. '3.11'.
  #
  # CPython's Limited API guards expect a hexadecimal like '0x030b0000', where:
  #
  # * '0x'   = prefix for a hex literal
  # * '03'   = major version: 3
  # * '0b'   = minor version (0x0b = 11)
  # * '0000' = other version components that can be ignored (limited API is versioned by Python
  #   major.minor)
  #
  # docs: https://docs.python.org/3/c-api/stable.html#c.Py_LIMITED_API
  #
  string(REPLACE "." ";" _sabi_parts "${SKBUILD_SABI_VERSION}")
  list(GET _sabi_parts 0 _sabi_major)
  list(GET _sabi_parts 1 _sabi_minor)
  math(EXPR _sabi_major_minor_hex "${_sabi_major} * 256 * 65536 + ${_sabi_minor} * 65536"
       OUTPUT_FORMAT HEXADECIMAL
  )

  # CPython source code pads '3' to '03' so all version components use 2 digits. Mirror that.
  string(REGEX REPLACE "^0x" "0x0" _sabi_major_minor_hex "${_sabi_major_minor_hex}")
  target_compile_definitions(${target} PRIVATE "Py_LIMITED_API=${_sabi_major_minor_hex}")
endfunction()
