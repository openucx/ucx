# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# cmake-format: on
# =================================================================================

find_package(Threads REQUIRED)

# ##################################################################################################
# * compiler function -----------------------------------------------------------------------------

add_custom_command(
  OUTPUT UCXX_EXAMPLES
  COMMAND echo Running examples
  COMMAND mkdir -p results
  VERBATIM
  COMMENT "Running ucxx examples."
  USES_TERMINAL
)

# This function takes in an example name and example source and handles setting all of the
# associated properties and linking to build the example
function(ConfigureBench CMAKE_BENCH_NAME)
  add_executable(${CMAKE_BENCH_NAME} ${ARGN})
  set_target_properties(
    ${CMAKE_BENCH_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${UCXX_BINARY_DIR}/examples>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
  )
  target_link_libraries(${CMAKE_BENCH_NAME} PRIVATE ucxx $<TARGET_NAME_IF_EXISTS:conda_env>)
  if(UCXX_ENABLE_CCCL)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(${CMAKE_BENCH_NAME} PRIVATE CUDA::cudart_static)
  endif()
  add_custom_command(
    OUTPUT UCXX_EXAMPLES
    COMMAND ${CMAKE_BENCH_NAME}
    APPEND
    COMMENT "Adding ${CMAKE_BENCH_NAME}"
  )

  install(
    TARGETS ${CMAKE_BENCH_NAME}
    COMPONENT examples
    DESTINATION bin/examples/libucxx
    EXCLUDE_FROM_ALL
  )
endfunction()

# ##################################################################################################
# * basic example ---------------------------------------------------------------------------------
ConfigureBench(ucxx_example_basic basic.cpp)

add_custom_target(
  run_examples
  DEPENDS UCXX_EXAMPLES
  COMMENT "Custom command for running ucxx examples."
)
