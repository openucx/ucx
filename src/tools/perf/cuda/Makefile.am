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
libucx_perftest_cuda_la_SOURCES  = cuda_alloc.c ucp_cuda_impl.c

if HAVE_NVCC

NVCC_EXTRA_FLAGS = $(CUDA_CPPFLAGS)

libucx_perftest_cuda_la_LIBADD  += \
    $(abs_top_builddir)/src/tools/perf/lib/libucxperf.la \
    -lstdc++

libucx_perftest_cuda_la_SOURCES += \
    ucp_cuda_kernel_bw.cu \
    ucp_cuda_kernel_bw_thread_fc.cu \
    ucp_cuda_kernel_bw_thread_nofc.cu \
    ucp_cuda_kernel_bw_warp_fc.cu \
    ucp_cuda_kernel_bw_warp_nofc.cu \
    ucp_cuda_kernel_latency.cu \
    ucp_cuda_kernel_latency_thread.cu \
    ucp_cuda_kernel_latency_warp.cu \
    ucp_cuda_kernel_wait.cu \
    ucp_cuda_host.cu

noinst_HEADERS = \
    cuda_common.h \
    cuda_context.h \
    cuda_kernel.cuh \
    ucp_cuda_impl.h \
    ucp_cuda_kernel_bw.cuh \
    ucp_cuda_kernel_bw_dispatch.cuh \
    ucp_cuda_kernel_bw_impl.cuh \
    ucp_cuda_kernel_common.cuh \
    ucp_cuda_kernel_latency.cuh \
    ucp_cuda_kernel_latency_dispatch.cuh \
    ucp_cuda_kernel_latency_impl.cuh \
    ucp_cuda_kernel_wait.cuh

include $(top_srcdir)/config/cuda.am
endif # HAVE_NVCC

include $(top_srcdir)/config/module.am

endif # HAVE_CUDA
