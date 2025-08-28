#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_NVCC
#TODO: if HAVE_GPUNETIO

include $(top_srcdir)/config/cuda.am

lib_LTLIBRARIES = libperf_gdaki.la

libperf_gdaki_la_SOURCES = \
    gdaki_mem.cc

-include gdaki_mem.lo.d

libperf_gdaki_la_CPPFLAGS = $(BASE_CPPFLAGS) $(CUDA_CPPFLAGS) -I$(top_srcdir)/src
libperf_gdaki_la_LDFLAGS  = $(CUDA_LDFLAGS) -version-info $(SOVERSION)
libperf_gdaki_la_LIBADD   = $(CUDART_LIBS)

EXTRA_DIST = gdaki_mem.h

# endif # HAVE_GPUNETIO
endif # HAVE_NVCC
