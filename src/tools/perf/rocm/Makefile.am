#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if HAVE_ROCM

module_LTLIBRARIES               = libucx_perftest_rocm.la
libucx_perftest_rocm_la_CPPFLAGS = $(BASE_CPPFLAGS) $(ROCM_CPPFLAGS) -D__HIP_PLATFORM_HCC__= -I/opt/rocm/hip/include -I/opt/rocm/hcc/include -I/opt/rocm/hsa/include
libucx_perftest_rocm_la_CFLAGS   = $(BASE_CFLAGS) $(ROCM_CFLAGS) -D__HIP_PLATFORM_HCC__= -I/opt/rocm/hip/include -I/opt/rocm/hcc/include -I/opt/rocm/hsa/include
libucx_perftest_rocm_la_LDFLAGS  = $(ROCM_LDFLAGS) -version-info $(SOVERSION)
libucx_perftest_rocm_la_SOURCES  = rocm_alloc.c

include $(top_srcdir)/config/module.am

endif
