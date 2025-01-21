#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2001-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# See file LICENSE for terms.
#

if HAVE_CUDA

module_LTLIBRARIES               = libucx_perftest_cuda.la
libucx_perftest_cuda_la_CPPFLAGS = $(BASE_CPPFLAGS) $(CUDA_CPPFLAGS)
libucx_perftest_cuda_la_CFLAGS   = $(BASE_CFLAGS) $(CUDA_CFLAGS) $(LT_CFLAGS)
libucx_perftest_cuda_la_LDFLAGS  = $(CUDA_LDFLAGS) -version-info $(SOVERSION)
libucx_perftest_cuda_la_LIBADD   = $(CUDART_LIBS)
libucx_perftest_cuda_la_SOURCES  = cuda_alloc.c

include $(top_srcdir)/config/module.am

endif
