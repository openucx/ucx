#
# Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if HAVE_HIP

module_LTLIBRARIES               = libucx_perftest_rocm.la
libucx_perftest_rocm_la_CPPFLAGS = $(BASE_CPPFLAGS) $(HIP_CPPFLAGS)
libucx_perftest_rocm_la_CFLAGS   = $(BASE_CFLAGS) $(HIP_CFLAGS)
libucx_perftest_rocm_la_LDFLAGS  = $(HIP_LDFLAGS) $(HIP_LIBS) -version-info $(SOVERSION)
libucx_perftest_rocm_la_SOURCES  = rocm_alloc.c

include $(top_srcdir)/config/module.am

endif
