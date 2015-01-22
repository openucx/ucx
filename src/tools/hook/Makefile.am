#
# Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
#
# $COPYRIGHT$
# $HEADER$
#

lib_LTLIBRARIES = libucxmalloc.la

#
# We an include headers from UCS, but avoid linking with UCS for now, because
# it has too many library constructors. Which means we cannot use most of its 
# services (e.g logging).
#
AM_CPPFLAGS           = \
    -I$(abs_top_srcdir)/src \
    -I$(abs_top_builddir)/src

libucxmalloc_la_SOURCES = \
	ucxmalloc.c

