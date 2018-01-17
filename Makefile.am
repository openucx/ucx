#
# Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
# Copyright (C) The University of Tennessee and The University
#               of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

# Build . before src so that our all-local and clean-local hooks kicks in at
# the right time.

EXTRA_DIST =
ACLOCAL_AMFLAGS = -I config/m4

noinst_HEADERS = src/uct/api/uct.h src/uct/api/uct_def.h src/uct/api/tl.h
doxygen_doc_files = $(noinst_HEADERS)

doc_dir = $(pkgdatadir)/doc

if !DOCS_ONLY
perftest_dir = $(pkgdatadir)/perftest
dist_perftest__DATA = contrib/ucx_perftest_config/msg_pow2 contrib/ucx_perftest_config/README contrib/ucx_perftest_config/test_types contrib/ucx_perftest_config/transports

SUBDIRS = \
	src/ucm \
	src/ucs \
	src/uct \
	src/ucp \
	src/tools/info \
	src/tools/perf \
	src/tools/profile \
	test/apps \
	test/examples

if HAVE_GTEST
SUBDIRS += test/gtest
endif

if HAVE_MPICC
SUBDIRS += test/mpi
endif

if HAVE_JAVA
SUBDIRS += bindings/java/src/main/native
endif

EXTRA_DIST += config/m4/java.m4
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
EXTRA_DIST += debian
EXTRA_DIST += ucx.pc.in
EXTRA_DIST += LICENSE
endif #!DOCS_ONLY
EXTRA_DIST += doc/uml/uct.dot

include $(srcdir)/doc/doxygen/doxygen.am

.PHONY: docs docs-clean

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = ucx.pc

DOCLIST = doc/doxygen/doxygen-doc/ucx.tag

FORMAT = pdf
DOT_CLEANFILES =

if HAVE_GTEST
gtest:
	@make -C test/gtest test
endif

if HAVE_DOT
DOCLIST += doc/uml/uml.tag doc/uml/uct.$(FORMAT) doc/uml/ucp.$(FORMAT)
DOT_CLEANFILES += doc/uml/uml.tag doc/uml/uct.$(FORMAT) doc/uml/ucp.$(FORMAT)
endif

docs: $(DOCLIST)

docs-clean:
	$(RM) $(DX_CLEANFILES)
	$(RM) $(DOT_CLEANFILES)

doc/doxygen/doxygen-doc/ucx.tag: $(doxygen_doc_files) doxygen-doc

doc/uml/uml.tag:
	mkdir -p doc/uml
	echo `date` > $@

.dot.pdf:
	dot -T pdf -o $@ $<

MOSTLYCLEANFILES = $(DX_CLEANFILES) $(DOT_CLEANFILES)
