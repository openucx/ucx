#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_NVCC
if HAVE_GPUNETIO

lib_LTLIBRARIES = libperf_gdaki.la

libperf_gdaki_la_SOURCES = \
    gdaki_benchmark.h \
    libperf_gdaki.h \
    gdaki_mem_handle.h \
    gdaki_mem_handle.cu \
    bw_benchmark.cu \
    lat_benchmark.cu

-include gdaki_mem_handle.lo.d
-include bw_benchmark.lo.d
-include lat_benchmark.lo.d

noinst_HEADERS = gdaki_benchmark.h

# Compile CUDA files with NVCC wrapper
SUFFIXES = .cu
# TODO: define rule for .cu.lo, .cu.o and improve this rule to use predefined opts.
.cu.lo:
	$(LIBTOOL) --tag=CXX --mode=compile $(NVCC_WRAP) "$(NVCC)" $(NVCC_FLAGS) \
		-c $< -MD -MP -MT $@ -MF $@.d -o $@ -DHAVE_CONFIG_H $(BASE_CPPFLAGS) $(GPUNETIO_CFLAGS)

libperf_gdaki_la_LDFLAGS = -version-info 0:0:0

libperf_gdaki_la_CPPFLAGS = $(BASE_CPPFLAGS) -I$(top_srcdir)/src

libperf_gdaki_la_LIBADD = -lcuda -lcudart -lstdc++

if HAVE_GDR_COPY
libperf_gdaki_la_LDFLAGS += $(GDR_COPY_LDFLAGS)
libperf_gdaki_la_CPPFLAGS += $(GDR_COPY_CPPFLAGS)
libperf_gdaki_la_LIBADD += $(GDR_COPY_LIBS)
endif

# Make sure wrapper script is executable
EXTRA_DIST = gdaki_benchmark.h gdaki_mem_handle.h gdaki_mem_handle.cu bw_benchmark.cu lat_benchmark.cu nvcc_wrap.sh

endif # HAVE_GPUNETIO
endif # HAVE_NVCC
