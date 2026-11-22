# ======================================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# cmake-format: on
# ======================================================================================================

# ##################################################################################################
# * compiler function -----------------------------------------------------------------------------

# This function takes in a test name and test source and handles setting all of the associated
# properties and linking to build the test
function(ConfigureTest CMAKE_TEST_NAME)
  add_executable(${CMAKE_TEST_NAME} ${ARGN})
  set_target_properties(
    ${CMAKE_TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${UCXX_BINARY_DIR}/gtests>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
  )

  target_include_directories(
    ${CMAKE_TEST_NAME} PUBLIC "$<BUILD_INTERFACE:${UCXX_SOURCE_DIR}>"
                              "$<BUILD_INTERFACE:${UCXX_SOURCE_DIR}/src>"
  )

  target_link_libraries(
    ${CMAKE_TEST_NAME} PRIVATE ucxx GTest::gmock_main GTest::gtest_main
                               $<TARGET_NAME_IF_EXISTS:conda_env>
  )
  if(UCXX_ENABLE_CCCL)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(${CMAKE_TEST_NAME} PRIVATE CUDA::cudart_static)
  endif()
  if(UCXX_TESTS_ENABLE_RMM)
    target_compile_definitions(${CMAKE_TEST_NAME} PRIVATE UCXX_TESTS_ENABLE_RMM)
    target_link_libraries(${CMAKE_TEST_NAME} PRIVATE rmm::rmm)
  endif()
  add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
  install(
    TARGETS ${CMAKE_TEST_NAME}
    COMPONENT testing
    DESTINATION bin/gtests/libucxx
    EXCLUDE_FROM_ALL
  )
endfunction()

# ##################################################################################################
# test sources ##################################################################################
# ##################################################################################################

# ##################################################################################################
# * ucxx tests ------------------------------------------------------------------------------------
ConfigureTest(
  UCXX_TEST buffer.cpp component_builders.cpp config.cpp context.cpp endpoint.cpp header.cpp
  listener.cpp request.cpp request_builders.cpp rma.cpp utils.cpp worker.cpp
)

# ##################################################################################################
# enable testing ################################################################################
# ##################################################################################################

enable_testing()
