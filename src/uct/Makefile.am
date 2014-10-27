#
# Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
#
# $COPYRIGHT$
# $HEADER$
#

lib_LTLIBRARIES     = libuct.la

libuct_la_CPPFLAGS = -I$(abs_top_srcdir)/src -I$(abs_top_builddir)/src
libuct_la_LDFLAGS  = -ldl -version-info $(SOVERSION)
libuct_la_LIBADD   = $(LIBM)
libuct_ladir       = $(includedir)

nobase_dist_libuct_la_HEADERS = \
	api/tl.h \
	api/uct_def.h \
	api/uct.h \
	api/version.h

noinst_HEADERS =

libuct_la_SOURCES = \
	tl/tl.c
