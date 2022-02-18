#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_CMA

module_LTLIBRARIES     = libuct_cma.la
libuct_cma_la_CFLAGS   = $(BASE_CFLAGS)
libuct_cma_la_CPPFLAGS = $(BASE_CPPFLAGS)
libuct_cma_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                         $(top_builddir)/src/uct/libuct.la
libuct_cma_la_LDFLAGS  = -version-info $(SOVERSION)

noinst_HEADERS = \
	cma_iface.h \
	cma_ep.h \
	cma_md.h

libuct_cma_la_SOURCES = \
	cma_iface.c \
	cma_ep.c \
	cma_md.c

PKG_CONFIG_NAME=cma

include $(top_srcdir)/config/module.am
include $(top_srcdir)/config/module-pkg-config.am

endif
