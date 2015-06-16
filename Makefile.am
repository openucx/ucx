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
doxygen_doc_files = $(noinst_HEADERS)

docdir = $(pkgdatadir)/doc
dist_doc_DATA = README

SUBDIRS = \
	src/ucs \
	src/uct \
	src/ucp \
	src/tools/hook \
	src/tools/info \
	src/tools/perf \
	test/examples

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
EXTRA_DIST += contrib/buildrpm.sh
EXTRA_DIST += contrib/ucx_perftest_config/msg_pow2
EXTRA_DIST += contrib/ucx_perftest_config/README
EXTRA_DIST += contrib/ucx_perftest_config/test_types
EXTRA_DIST += contrib/ucx_perftest_config/transports

include $(srcdir)/doc/doxygen/doxygen.am

.PHONY: docs docs-clean

docs: doc/doxygen/doxygen-doc/ucx.tag

docs-clean:
	$(RM) $(DX_CLEANFILES)

doc/doxygen/doxygen-doc/ucx.tag: $(doxygen_doc_files) doxygen-doc

MOSTLYCLEANFILES = $(DX_CLEANFILES)
