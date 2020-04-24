#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# Copyright (C) ARM Ltd. 2017.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

SUBDIRS = profiling

if HAVE_GLIBCXX_NOTHROW
SUBDIRS += iodemo
endif

if HAVE_CXX11
SUBDIRS += sockaddr
endif

noinst_PROGRAMS = \
	test_ucp_dlopen \
	test_ucs_dlopen \
	test_link_map \
	test_dlopen_cfg_print

objdir = $(shell sed -n -e 's/^objdir=\(.*\)$$/\1/p' $(LIBTOOL))

test_ucs_dlopen_SOURCES  = test_ucs_dlopen.c
test_ucs_dlopen_CPPFLAGS = $(BASE_CPPFLAGS) \
                           -DLIB_PATH=$(abs_top_builddir)/src/ucs/$(objdir)/libucs.so
test_ucs_dlopen_CFLAGS   = $(BASE_CFLAGS)
test_ucs_dlopen_LDADD    = -ldl

test_ucp_dlopen_SOURCES  = test_ucp_dlopen.c
test_ucp_dlopen_CPPFLAGS = $(BASE_CPPFLAGS) \
                           -DLIB_PATH=$(abs_top_builddir)/src/ucp/$(objdir)/libucp.so
test_ucp_dlopen_CFLAGS   = $(BASE_CFLAGS)
test_ucp_dlopen_LDADD    = -ldl

test_link_map_SOURCES  = test_link_map.c
test_link_map_CPPFLAGS = $(BASE_CPPFLAGS)
test_link_map_CFLAGS   = $(BASE_CFLAGS)
test_link_map_LDADD    = -ldl $(top_builddir)/src/ucp/libucp.la

test_dlopen_cfg_print_SOURCES  = test_dlopen_cfg_print.c
test_dlopen_cfg_print_CPPFLAGS = $(BASE_CPPFLAGS) -g \
	-DUCS_LIB_PATH=$(abs_top_builddir)/src/ucs/$(objdir)/libucs.so \
	-DUCT_LIB_PATH=$(abs_top_builddir)/src/uct/$(objdir)/libuct.so
test_dlopen_cfg_print_CFLAGS   = $(BASE_CFLAGS)
test_dlopen_cfg_print_LDADD    = -ldl

if HAVE_TCMALLOC
noinst_PROGRAMS       += test_tcmalloc
test_tcmalloc_SOURCES  = test_tcmalloc.c
test_tcmalloc_CPPFLAGS = $(BASE_CPPFLAGS)
test_tcmalloc_CFLAGS   = $(BASE_CFLAGS)
test_tcmalloc_LDADD    = -ldl $(TCMALLOC_LIB) \
                          $(top_builddir)/src/ucp/libucp.la
endif
