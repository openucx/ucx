#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

noinst_PROGRAMS = ucx_profiling

ucx_profiling_SOURCES  = ucx_profiling.c
ucx_profiling_CFLAGS   = $(BASE_CFLAGS)
ucx_profiling_CPPFLAGS = $(BASE_CPPFLAGS)
ucx_profiling_LDADD    = $(top_builddir)/src/ucs/libucs.la
ucx_profiling_LDFLAGS  = -lm
