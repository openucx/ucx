#
# Copyright (C) SambaNova Systems, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_RDU

module_LTLIBRARIES     = libucm_rdu.la
libucm_rdu_la_CPPFLAGS = $(BASE_CPPFLAGS)
libucm_rdu_la_CFLAGS   = $(BASE_CFLAGS) $(LT_CFLAGS)
libucm_rdu_la_LIBADD   = ../libucm.la $(RDU_LIBS)
libucm_rdu_la_LDFLAGS  = $(UCM_MODULE_LDFLAGS) \
                         -version-info $(SOVERSION) $(UCX_LT_RELEASE)

noinst_HEADERS = \
	rdumem.h

libucm_rdu_la_SOURCES = \
	rdumem.c

include $(top_srcdir)/config/module.am

endif
