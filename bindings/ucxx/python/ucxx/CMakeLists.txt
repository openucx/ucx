# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
# cmake-format: on
# =================================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

file(READ "${CMAKE_CURRENT_LIST_DIR}/../../VERSION" _version_contents)
if(_version_contents MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+).*$")
  set(ucxx_version "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
else()
  string(REPLACE "\n" "\n  " _version_contents_formatted "${_version_contents}")
  message(
    FATAL_ERROR
      "Could not determine ucxx version. Contents of VERSION file:\n  ${_version_contents_formatted}"
  )
endif()

include(../../cmake/rapids_config.cmake)

project(
  ucxx-python
  VERSION ${ucxx_version}
  LANGUAGES CXX
)

option(FIND_UCXX_PYTHON
       "Search for existing UCXX Python library installations before defaulting to local files" OFF
)
option(UCXX_BUILD_TESTS "Configure CMake to build Cython tests" OFF)

# If the user requested it we attempt to find UCXX.
if(FIND_UCXX_PYTHON)
  find_package(ucxx-python ${ucxx_version} REQUIRED)
else()
  set(ucxx-python_FOUND OFF)
endif()

include(rapids-cython-core)

if(NOT ucxx-python_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)

  set(_exclude_from_all "")

  add_subdirectory(../../cpp/python ucxx-python ${_exclude_from_all})
endif()

rapids_cython_init()

find_package(Python3 REQUIRED COMPONENTS Development.Embed)
add_subdirectory(ucxx/examples)
add_subdirectory(ucxx/_lib)
if(UCXX_BUILD_TESTS)
  add_subdirectory(ucxx/_lib/tests_cython)
endif()

if(NOT ucxx-python_FOUND)
  rapids_cython_add_rpath_entries(TARGET ucxx_python PATHS ${CMAKE_INSTALL_LIBDIR})
endif()
