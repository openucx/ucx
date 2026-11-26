# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
# cmake-format: on
# =================================================================================

set(cython_sources python_future_task_app.pyx)
set(linked_libraries ucxx::ucxx ucxx::python)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" ASSOCIATED_TARGETS ucxx ucxx_python COMPONENT examples
)

target_include_directories(python_future_task_app PRIVATE ".")
