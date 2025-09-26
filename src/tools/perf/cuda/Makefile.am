#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
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

if HAVE_NVCC
if HAVE_GPUNETIO # temporarily all device code depends on doca

NVCC_EXTRA_FLAGS = $(CUDA_CPPFLAGS) $(GPUNETIO_CFLAGS)

libucx_perftest_cuda_la_LIBADD  += \
    $(abs_top_builddir)/src/tools/perf/lib/libucxperf.la \
    -lstdc++

libucx_perftest_cuda_la_SOURCES += \
    ucp_cuda_kernel.cu

noinst_HEADERS = \
    cuda_common.h \
    cuda_kernel.cuh

include $(top_srcdir)/config/cuda.am
endif # HAVE_GPUNETIO
endif # HAVE_NVCC

include $(top_srcdir)/config/module.am

endif # HAVE_CUDA
