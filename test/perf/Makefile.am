#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
#
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# $COPYRIGHT$
# $HEADER$
#

if HAVE_MPI
ucx_perftest_CC = $(MPICC)
else
ucx_perftest_CC = $(CC)
endif
	
bin_PROGRAMS          = ucx_perftest
noinst_LTLIBRARIES    = libucxperf.la
AM_CPPFLAGS           = \
    -I$(abs_top_srcdir)/src \
    -I$(abs_top_builddir)/src \
    $(RTE_CPPFLAGS)

noinst_HEADERS        = libperf.h
libucxperf_la_SOURCES = libperf.c
libucxperf_la_LDFLAGS   = \
     $(RTE_LDFLAGS)
libucxperf_la_LIBADD   = \
     $(abs_top_builddir)/src/uct/libuct.la \
     $(abs_top_builddir)/src/ucs/libucs.la

ucx_perftest_SOURCES  = perftest.c
ucx_perftest_LDFLAGS    = \
     $(RTE_LDFLAGS)
ucx_perftest_LDADD    = \
     $(abs_top_builddir)/src/uct/libuct.la \
     $(abs_top_builddir)/src/ucs/libucs.la \
     libucxperf.la
ucx_perftest_SOURCE_FILES = \
	$(patsubst %, $(srcdir)/%, $(ucx_perftest_SOURCES))

#
# Add custom rules to compile and link ucx_perftest with MPICC
#
perftest.o: $(ucx_perftest_SOURCE_FILES)
	$(AM_V_CC)
	@$(LIBTOOL) $(AM_V_lt) --mode=compile --tag=CC $(ucx_perftest_CC) \
		$(DEFS) -I$(top_builddir) $(AM_CPPFLAGS) $(ucx_perftest_CPPFLAGS) \
		$(ucx_perftest_CFLAGS) $(ucx_perftest_SOURCE_FILES) -c -o $@

ucx_perftest$(EXEEXT): perftest.o $(ucx_perftest_LDADD)
	$(AM_V_CCLD)
	@$(LIBTOOL) $(AM_V_lt) --mode=link --tag=CC $(ucx_perftest_CC) \
		perftest.o $(ucx_perftest_LDFLAGS) $(ucx_perftest_LDADD) -o $@

if HAVE_MPIRUN
.PHONY: ucx test help

MPI_EXTRA =
MPI_ARGS  = -n 2 -map-by node -mca pml ob1 -mca btl self,tcp,sm $(MPI_EXTRA)

ucx:
	$(MAKE) -C $(top_builddir)

test: ucx ucx_perftest
	$(MPIRUN) $(MPI_ARGS) ucx_perftest$(EXEEXT) $(TEST_ARGS)

help:
	@echo
	@echo "Targets:"
	@echo "  test       : Run performance test."
	@echo
	@echo "Evnironment variables:"
	@echo "  MPI_ARGS   : Arguments for mpirun (\"$(MPI_ARGS)\")"
	@echo "  TEST_ARGS  : Arguments for performance test (\"$(TEST_ARGS)\")"
	@echo

endif
