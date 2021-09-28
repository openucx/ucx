#
# Copyright (C) NVIDIA Corporation. 2021.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

cmake_minimum_required(VERSION 2.8)

project(test_ucx)

find_package(UCX REQUIRED COMPONENTS ucp uct ucs)

add_executable(test_ucp test_ucp.c)
target_link_libraries(test_ucp ucx::ucp)

add_executable(test_uct test_uct.c)
target_link_libraries(test_uct ucx::uct ucx::ucs)
