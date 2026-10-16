#
# Copyright (C) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_MUSA
module_LTLIBRARIES      = libucm_musa.la
libucm_musa_la_CPPFLAGS = $(BASE_CPPFLAGS) $(MUSA_CPPFLAGS)
libucm_musa_la_CFLAGS   = $(BASE_CFLAGS) $(MUSA_CFLAGS) $(LT_CFLAGS)
libucm_musa_la_LIBADD   = ../libucm.la $(MUSA_LIBS)
libucm_musa_la_LDFLAGS  = $(UCM_MODULE_LDFLAGS) \
                          $(patsubst %, -Xlinker %, $(MUSA_LDFLAGS)) \
                          -version-info $(SOVERSION) $(UCX_LT_RELEASE)

libucm_musa_la_SOURCES = \
	musamem.c

include $(top_srcdir)/config/module.am
endif
