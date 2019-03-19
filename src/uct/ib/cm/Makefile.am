#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_TL_CM

module_LTLIBRARIES       = libuct_ib_cm.la
libuct_ib_cm_la_CPPFLAGS = $(BASE_CPPFLAGS) $(IBVERBS_CPPFLAGS)
libuct_ib_cm_la_CFLAGS   = $(BASE_CFLAGS)
libuct_ib_cm_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                           $(top_builddir)/src/uct/ib/libuct_ib.la
libuct_ib_cm_la_LDFLAGS  = $(IBVERBS_LDFLAGS) $(IBCM_LIBS) -version-info $(SOVERSION)

noinst_HEADERS = \
	cm.h

libuct_ib_cm_la_SOURCES = \
	cm_iface.c \
	cm_ep.c
	
include $(top_srcdir)/config/module.am

endif
