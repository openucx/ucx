#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_NVCC

include $(top_srcdir)/config/cuda.am

lib_LTLIBRARIES = libperf_device.la

libperf_device_la_SOURCES = \
    device_mem.c

-include device_mem.lo.d

libperf_device_la_CPPFLAGS = $(BASE_CPPFLAGS) $(CUDA_CPPFLAGS) -I$(top_srcdir)/src
libperf_device_la_CFLAGS   = $(BASE_CFLAGS) $(CUDA_CFLAGS)
libperf_device_la_LDFLAGS  = $(CUDA_LDFLAGS) -version-info $(SOVERSION)
libperf_device_la_LIBADD   = $(CUDART_LIBS)

EXTRA_DIST = libperf_device.h device_mem.h ucp_device_tests.h

endif # HAVE_NVCC
