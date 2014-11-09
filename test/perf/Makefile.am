#
# Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
#
# $COPYRIGHT$
# $HEADER$
#

bin_PROGRAMS = ucx_perftest

ucx_perftest_SOURCES = perftest.c
ucx_perftest_LDADD   = $(abs_top_builddir)/src/uct/libuct.la
ucx_perftest_CPPFLAGS = \
	-I$(abs_top_srcdir)/src \
	-I$(abs_top_builddir)/src
