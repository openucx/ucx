#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# Copyright (C) ARM Ltd. 2017.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

noinst_PROGRAMS = \
	test_dlopen


objdir = $(shell sed -n -e 's/^objdir=\(.*\)$$/\1/p' $(LIBTOOL))

test_dlopen_SOURCES  = dlopen.c
test_dlopen_CPPFLAGS = $(BASE_CPPFLAGS) -g -DUCP_LIB_PATH=$(abs_top_builddir)/src/ucp/$(objdir)/libucp.so
test_dlopen_CFLAGS   = $(BASE_CFLAGS)
test_dlopen_LDADD    = -ldl

if HAVE_PROFILING
noinst_PROGRAMS += \
	test_profiling

test_profiling_SOURCES  = profiling.c
test_profiling_LDADD    = $(top_builddir)/src/ucs/libucs.la -lm
test_profiling_CPPFLAGS = $(BASE_CPPFLAGS) -g -I$(top_srcdir)/src
test_profiling_CFLAGS   = $(BASE_CFLAGS)
endif
