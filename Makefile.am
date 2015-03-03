#
# Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
#
# $COPYRIGHT$
# $HEADER$
#

# Build . before src so that our all-local and clean-local hooks kicks in at
# the right time.

ACLOCAL_AMFLAGS = -I config/m4

noinst_HEADERS = src/uct/api/uct.h src/uct/api/uct_def.h src/uct/api/tl.h

SUBDIRS = \
	src/ucs \
	src/uct \
	src/ucp \
	src/tools/hook \
	src/tools/info \
	src/tools/perf

if HAVE_GTEST
SUBDIRS += test/gtest
endif

EXTRA_DIST =
EXTRA_DIST += config/m4/gtest.m4
EXTRA_DIST += config/m4/ucs.m4
EXTRA_DIST += config/m4/ib.m4
EXTRA_DIST += config/m4/sysdep.m4
EXTRA_DIST += contrib/configure-devel
EXTRA_DIST += contrib/configure-release
EXTRA_DIST += contrib/configure-prof

include $(srcdir)/doc/doxygen/doxygen.am

.PHONY: docs

docs: doc/doxygen/doxygen-doc/ucx.tag

doc/doxygen/doxygen-doc/ucx.tag: $(noinst_HEADERS) doxygen-doc

MOSTLYCLEANFILES = $(DX_CLEANFILES)
