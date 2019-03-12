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
libucm_rocm_la_LDFLAGS  = $(UCM_MODULE_LDFLAGS) $(ROCM_LDFLAGS) \
                          -version-info $(SOVERSION)

noinst_HEADERS = \
	rocmmem.h

libucm_rocm_la_SOURCES = \
	rocmmem.c

include $(top_srcdir)/config/module.am

endif
