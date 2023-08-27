#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2015-2017. ALL RIGHTS RESERVED.
# Copyright (C) The University of Tennessee and The University
#               of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
# Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

noinst_LTLIBRARIES     = libucxperf.la
libucxperf_la_CPPFLAGS = $(BASE_CPPFLAGS)
libucxperf_la_CFLAGS   = $(BASE_CFLAGS) $(OPENMP_CFLAGS)
libucxperf_la_CXXFLAGS = $(BASE_CXXFLAGS) $(OPENMP_CFLAGS)
libucxperf_la_LDFLAGS  = $(RTE_LDFLAGS) $(OPENMP_CFLAGS)
libucxperf_la_LIBADD   = \
	$(abs_top_builddir)/src/uct/libuct.la \
	$(abs_top_builddir)/src/ucp/libucp.la \
	$(abs_top_builddir)/src/ucs/libucs.la

# Add CUDA dependencies
if HAVE_CUDA
libucxperf_la_CPPFLAGS += $(CUDA_CPPFLAGS)
libucxperf_la_CFLAGS   += $(CUDA_CFLAGS)
libucxperf_la_LDFLAGS  += $(CUDA_LDFLAGS) -version-info $(SOVERSION)
libucxperf_la_LIBADD   += $(CUDART_LIBS)
endif

# Add ROCM dependencies
if HAVE_HIP
libucxperf_la_CPPFLAGS += $(HIP_CPPFLAGS)
libucxperf_la_CFLAGS   += $(HIP_CFLAGS)
libucxperf_la_LDFLAGS  += $(HIP_LDFLAGS) $(HIP_LIBS) -version-info $(SOVERSION) \
                          $(patsubst %, -Xlinker %, -L$(ROCM_ROOT)/lib -rpath $(ROCM_ROOT)/hip/lib -rpath $(ROCM_ROOT)/lib) \
                          $(patsubst %, -Xlinker %, --enable-new-dtags) \
                          $(patsubst %, -Xlinker %, -rpath $(ROCM_ROOT)/lib64)
endif


# C-linkable C++ code - must override any inherited CXXFLAGS
CXXFLAGS              += -nostdlib $(PERF_LIB_CXXFLAGS) -std=c++11

noinst_HEADERS = \
	libperf_int.h

libucxperf_la_SOURCES = \
	libperf.c \
	libperf_memcpy.c \
	libperf_memory.c \
	libperf_thread.c \
	ucp_tests.cc \
	uct_tests.cc
