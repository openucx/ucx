#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
#
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_PROFILING
noinst_PROGRAMS = \
	profiling

profiling_SOURCES  = profiling.c
profiling_LDADD    = $(top_builddir)/src/ucs/libucs.la
profiling_CPPFLAGS = -g -I$(top_srcdir)/src
endif
