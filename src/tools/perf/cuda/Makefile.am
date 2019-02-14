#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if HAVE_CUDA

module_LTLIBRARIES               = libucx_perftest_cuda.la
libucx_perftest_cuda_la_CPPFLAGS = $(BASE_CPPFLAGS) $(CUDA_CPPFLAGS)
libucx_perftest_cuda_la_CFLAGS   = $(BASE_CFLAGS) $(CUDA_CFLAGS)
libucx_perftest_cuda_la_LDFLAGS  = $(CUDA_LDFLAGS) -version-info $(SOVERSION)
libucx_perftest_cuda_la_SOURCES  = cuda_alloc.c

include $(top_srcdir)/config/module.am

endif
