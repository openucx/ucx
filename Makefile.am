#
# Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

# Build . before src so that our all-local and clean-local hooks kicks in at
# the right time.

ACLOCAL_AMFLAGS = -I config/m4

noinst_HEADERS = src/uct/api/uct.h src/uct/api/uct_def.h src/uct/api/tl.h
doxygen_doc_files = $(noinst_HEADERS)

perftest_dir = $(pkgdatadir)/perftest
doc_dir = $(pkgdatadir)/doc

dist_perftest__DATA = contrib/ucx_perftest_config/msg_pow2 contrib/ucx_perftest_config/README contrib/ucx_perftest_config/test_types contrib/ucx_perftest_config/transports
dist_doc__DATA = README

SUBDIRS = \
	src/ucs \
	src/uct \
	src/ucp \
	src/tools/hook \
	src/tools/info \
	src/tools/perf \
	test/examples

if BUILD_TCMALLOC
SUBDIRS += src/tools/hook/gperftools
endif

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
EXTRA_DIST += doc/uml/uct.dot

include $(srcdir)/doc/doxygen/doxygen.am

.PHONY: docs docs-clean

DOCLIST = doc/doxygen/doxygen-doc/ucx.tag

FORMAT = pdf
DOT_CLEANFILES =

if HAVE_DOT
DOCLIST += doc/uml/uct.$(FORMAT)
DOT_CLEANFILES += doc/uml/uct.$(FORMAT)
endif

docs: $(DOCLIST)

docs-clean:
	$(RM) $(DX_CLEANFILES)
	$(RM) $(DOT_CLEANFILES)

doc/doxygen/doxygen-doc/ucx.tag: $(doxygen_doc_files) doxygen-doc

doc/uml/uct.$(FORMAT):
	mkdir -p $(abs_top_builddir)/doc/uml
	dot -T $(FORMAT) -o $(abs_top_builddir)/doc/uml/uct.pdf $(abs_top_srcdir)/doc/uml/uct.dot

MOSTLYCLEANFILES = $(DX_CLEANFILES) $(DOT_CLEANFILES)
