#
# Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if HAVE_HIP

module_LTLIBRARIES               = libucx_perftest_rocm.la
libucx_perftest_rocm_la_CPPFLAGS = $(BASE_CPPFLAGS) $(HIP_CPPFLAGS)
libucx_perftest_rocm_la_CFLAGS   = $(BASE_CFLAGS) $(HIP_CFLAGS)
libucx_perftest_rocm_la_LDFLAGS  = $(HIP_LDFLAGS) $(HIP_LIBS) -version-info $(SOVERSION) \
                          $(patsubst %, -Xlinker %, -L$(ROCM_ROOT)/lib -rpath $(ROCM_ROOT)/hip/lib -rpath $(ROCM_ROOT)/lib) \
                          $(patsubst %, -Xlinker %, --enable-new-dtags) \
                          $(patsubst %, -Xlinker %, -rpath $(ROCM_ROOT)/lib64)
libucx_perftest_rocm_la_SOURCES  = rocm_alloc.c

include $(top_srcdir)/config/module.am

endif
