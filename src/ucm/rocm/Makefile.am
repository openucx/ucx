#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if HAVE_ROCM

module_LTLIBRARIES      = libucm_rocm.la
libucm_rocm_la_CPPFLAGS = $(BASE_CPPFLAGS) $(ROCM_CPPFLAGS)
libucm_rocm_la_CFLAGS   = $(BASE_CFLAGS) $(ROCM_CFLAGS)
libucm_rocm_la_LIBADD   = ../libucm.la
libucm_rocm_la_LDFLAGS  = $(UCM_MODULE_LDFLAGS) \
                          $(ROCM_LDFLAGS) $(ROCM_LIBS) -version-info $(SOVERSION) \
                          $(patsubst %, -Xlinker %, -L$(ROCM_ROOT)/lib -rpath $(ROCM_ROOT)/hip/lib -rpath $(ROCM_ROOT)/lib) \
                          $(patsubst %, -Xlinker %, --enable-new-dtags) \
                          $(patsubst %, -Xlinker %, -rpath $(ROCM_ROOT)/lib64)

noinst_HEADERS = \
	rocmmem.h

libucm_rocm_la_SOURCES = \
	rocmmem.c

include $(top_srcdir)/config/module.am

endif
