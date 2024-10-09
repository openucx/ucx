#
# Copyright (C) Intel Corporation, 2023-2024. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if HAVE_ZE

module_LTLIBRARIES    = libucm_ze.la
libucm_ze_la_CPPFLAGS = $(BASE_CPPFLAGS) $(ZE_CPPFLAGS)
libucm_ze_la_CFLAGS   = $(BASE_CFLAGS) $(ZE_CFLAGS) $(LT_CFLAGS)
libucm_ze_la_LIBADD   = ../libucm.la $(ZE_LIBS)
libucm_ze_la_LDFLAGS  = $(UCM_MODULE_LDFLAGS) \
                          $(patsubst %, -Xlinker %, $(ZE_LDFLAGS)) \
                          -version-info $(SOVERSION)

noinst_HEADERS = \
	zemem.h

libucm_ze_la_SOURCES = \
	zemem.c

include $(top_srcdir)/config/module.am

endif
