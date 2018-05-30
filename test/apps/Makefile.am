#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# Copyright (C) ARM Ltd. 2017.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if HAVE_CXX11
SUBDIRS = sockaddr
endif

noinst_PROGRAMS = \
	test_dlopen


objdir = $(shell sed -n -e 's/^objdir=\(.*\)$$/\1/p' $(LIBTOOL))

test_dlopen_SOURCES  = test_dlopen.c
test_dlopen_CPPFLAGS = $(BASE_CPPFLAGS) -g -DUCP_LIB_PATH=$(abs_top_builddir)/src/ucp/$(objdir)/libucp.so
test_dlopen_CFLAGS   = $(BASE_CFLAGS)
test_dlopen_LDADD    = -ldl

if HAVE_TCMALLOC
noinst_PROGRAMS       += test_tcmalloc
test_tcmalloc_SOURCES  = test_tcmalloc.c
test_tcmalloc_CPPFLAGS = $(BASE_CPPFLAGS) -g
test_tcmalloc_CFLAGS   = $(BASE_CFLAGS)
test_tcmalloc_LDADD    = -ldl $(TCMALLOC_LIB) \
                          $(top_builddir)/src/ucp/libucp.la
endif
