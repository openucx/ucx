#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# $COPYRIGHT$
# $HEADER$
#

lib_LTLIBRARIES     = libucp.la

libucp_la_CFLAGS   =
libucp_la_LIBS     =
libucp_la_CPPFLAGS = -I$(abs_top_srcdir)/src -I$(abs_top_builddir)/src
libucp_la_LDFLAGS  = -ldl -version-info $(SOVERSION)
libucp_la_LIBADD   = ../ucs/libucs.la
libucp_ladir       = $(includedir)

nobase_dist_libucp_la_HEADERS = \
    api/ucp_def.h \
    api/ucp.h

noinst_HEADERS = 

libucp_la_SOURCES = \
    ucp.c 
