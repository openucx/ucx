#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2015-2017. ALL RIGHTS RESERVED.
# Copyright (C) The University of Tennessee and The University
#               of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
# Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if HAVE_MPICC
ucx_perftest_CC = $(MPICC)
else
ucx_perftest_CC = $(CC)
endif

# override CXXFLAGS to link C++ code with C linker
CXXFLAGS += -nostdlib -fno-exceptions -fno-rtti

perftestdir = $(pkgdatadir)/perftest
dist_perftest_DATA = \
	$(top_srcdir)/contrib/ucx_perftest_config/msg_pow2 \
	$(top_srcdir)/contrib/ucx_perftest_config/README \
	$(top_srcdir)/contrib/ucx_perftest_config/test_types \
	$(top_srcdir)/contrib/ucx_perftest_config/transports

bin_PROGRAMS          = ucx_perftest
noinst_LTLIBRARIES    = libucxperf.la

AM_CPPFLAGS           = \
    $(BASE_CPPFLAGS) \
    -I$(abs_top_srcdir)/src \
    -I$(abs_top_builddir)/src \
    $(RTE_CPPFLAGS)

noinst_HEADERS = \
	libperf.h \
	libperf_int.h
libucxperf_la_SOURCES = \
	libperf.c \
	uct_tests.cc \
	ucp_tests.cc
libucxperf_la_LDFLAGS = \
	$(RTE_LDFLAGS) \
	$(OPENMP_CFLAGS)
libucxperf_la_CXXFLAGS = \
	$(BASE_CXXFLAGS) \
	$(OPENMP_CFLAGS)
libucxperf_la_CFLAGS = \
	$(BASE_CFLAGS) \
	$(OPENMP_CFLAGS)
libucxperf_la_LIBADD = \
	$(abs_top_builddir)/src/uct/libuct.la \
	$(abs_top_builddir)/src/ucp/libucp.la \
	$(abs_top_builddir)/src/ucs/libucs.la
#libucxperf_la_LINK = $(CXX)

ucx_perftest_SOURCES  = perftest.c
ucx_perftest_LDFLAGS = \
	$(RTE_LDFLAGS) \
	$(OPENMP_CFLAGS)
ucx_perftest_CPPFLAGS = \
	$(BASE_CPPFLAGS)
ucx_perftest_CFLAGS = \
	$(BASE_CFLAGS) \
	$(OPENMP_CFLAGS)
ucx_perftest_LDADD = \
	$(abs_top_builddir)/src/uct/libuct.la \
	$(abs_top_builddir)/src/ucp/libucp.la \
	$(abs_top_builddir)/src/ucs/libucs.la \
	libucxperf.la
ucx_perftest_SOURCE_FILES = \
	$(patsubst %, $(srcdir)/%, $(ucx_perftest_SOURCES))

#
# Add custom rules to compile and link ucx_perftest with MPICC
#
perftest_COMPILE = \
	$(LIBTOOL) $(AM_V_lt) --mode=compile --tag=CC $(ucx_perftest_CC) \
	$(DEFS) -I. -I$(top_builddir) $(AM_CPPFLAGS) $(CPPFLAGS) $(ucx_perftest_CPPFLAGS) \
	$(AM_CFLAGS) $(CFLAGS) $(ucx_perftest_CFLAGS)

perftest.lo: $(ucx_perftest_SOURCE_FILES)
	$(AM_V_CC)
if AMDEP
	@$(perftest_COMPILE) -MT $@ -MD -MP -MF $(DEPDIR)/ucx_perftest-perftest.Po -c -o $@ $<
endif
	@$(perftest_COMPILE) $(ucx_perftest_SOURCE_FILES) -c -o $@

ucx_perftest$(EXEEXT): perftest.lo $(ucx_perftest_LDADD)
	$(AM_V_CCLD)
	@$(LIBTOOL) $(AM_V_lt) --mode=link --tag=CC $(ucx_perftest_CC) \
		perftest.lo $(LDFLAGS) $(ucx_perftest_LDFLAGS) $(ucx_perftest_LDADD) -o $@

if HAVE_MPIRUN
.PHONY: ucx test help

MPI_EXTRA =
MPI_ARGS  = -n 2 -map-by node -mca pml ob1 -mca btl self,tcp,sm $(MPI_EXTRA)

ucx:
	$(MAKE) -C $(top_builddir)

test: ucx ucx_perftest
	$(MPIRUN) $(MPI_ARGS) $(abs_builddir)/ucx_perftest$(EXEEXT) $(TEST_ARGS)

help:
	@echo
	@echo "Targets:"
	@echo "  test       : Run performance test."
	@echo
	@echo "Environment variables:"
	@echo "  MPI_ARGS   : Arguments for mpirun (\"$(MPI_ARGS)\")"
	@echo "  TEST_ARGS  : Arguments for performance test (\"$(TEST_ARGS)\")"
	@echo

endif
