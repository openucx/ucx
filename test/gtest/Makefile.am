#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
#
# Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
# $COPYRIGHT$
# $HEADER$
#

UCS_HANDLE_ERRORS ?= freeze
export UCS_HANDLE_ERRORS

GTEST_FILTER        ?= *
GTEST_EXTRA_ARGS    =
LAUNCHER            =
VALGRIND_EXTRA_ARGS =

GTEST_ARGS = \
	--gtest_filter=$(GTEST_FILTER) \
	$(GTEST_EXTRA_ARGS)

VALGRIND_ARGS = \
	--tool=memcheck \
	--leak-check=full \
	--track-origins=yes \
	--fair-sched=try \
	$(VALGRIND_EXTRA_ARGS)

noinst_PROGRAMS = gtest

gtestdir  = $(includedir)
gtest_LDADD = \
	$(top_builddir)/src/ucs/libucs.la \
	$(top_builddir)/src/uct/libuct.la \
	$(top_builddir)/src/ucp/libucp.la \
	$(top_builddir)/src/ucs/libucstest.la \
	$(top_builddir)/test/perf/libucxperf.la \
	$(GTEST_LIBS)

gtest_CPPFLAGS = \
	-I$(top_srcdir)/src \
	-I$(top_srcdir)/test \
	-I$(top_builddir)/src \
	-I$(top_srcdir)/test/gtest \
	$(GTEST_CPPFLAGS)

gtest_LDFLAGS  = $(GTEST_LDFLAGS) -no-install
gtest_CXXFLAGS = $(GTEST_CXXFLAGS) -g -O3

gtest_SOURCES = \
	uct/test_amo.cc \
	uct/test_amo_add.cc \
	uct/test_amo_cswap.cc \
	uct/test_amo_fadd.cc \
	uct/test_amo_swap.cc \
	uct/test_p2p_am.cc \
	uct/test_p2p_rma.cc \
	uct/test_uct_context.cc \
	uct/test_uct_perf.cc \
	uct/uct_p2p_test.cc \
	uct/uct_test.cc \
	ucp/ucp_test.cc \
	ucp/test_ucp_ep.cc \
	ucs/test_async.cc \
	ucs/test_class.cc \
	ucs/test_component.cc \
	ucs/test_config.cc \
	ucs/test_datatype.cc \
	ucs/test_debug.cc \
	ucs/test_memtrack.cc \
	ucs/test_instr.cc \
	ucs/test_math.cc \
	ucs/test_mpmc.cc \
	ucs/test_mpool.cc \
	ucs/test_stats.cc \
	ucs/test_sys.cc \
	ucs/test_time.cc \
	ucs/test_twheel.cc \
	ucs/test_frag_list.cc

noinst_HEADERS = \
	uct/uct_p2p_test.h \
	uct/uct_test.h \
	ucp/ucp_test.h

.PHONY: test test gdb valgrind fix_rpath ucx


all-local: gtest

ucx:
	$(MAKE) -C $(top_builddir)


help:
	@echo
	@echo "Targets:"
	@echo "  list          : List unit tests."
	@echo "  test          : Run unit tests."
	@echo "  test_gdb      : Run unit tests with GDB."
	@echo "  test_valgrind : Run unit tests with Valgrind."
	@echo
	@echo "Environment variables:"
	@echo "  GTEST_FILTER        : Unit tests filter (\"$(GTEST_FILTER)\")"
	@echo "  GTEST_EXTRA_ARGS    : Additional arguments for gtest (\"$(GTEST_EXTRA_ARGS)\")"
	@echo "  LAUNCHER            : Custom launcher for gtest executable (\"$(LAUNCHER)\")"
	@echo "  VALGRIND_EXTRA_ARGS : Additional arguments for Valgrind (\"$(VALGRIND_EXTRA_ARGS)\")"
	@echo

#
# List unit tests
#
list: gtest
	$(abs_builddir)/gtest --gtest_list_tests $(GTEST_ARGS)

#
# Run unit tests
#
test: ucx gtest
	@rm -f core.*
	$(LAUNCHER) $(abs_builddir)/gtest $(GTEST_ARGS)
	
#
# Run unit tests with GDB
#
test_gdb: ucx gtest
	echo -e 'r\nq' > .gdbcommands
	env UCS_HANDLE_ERRORS=none \
		UCS_GDB_PATH="" \
		gdb -x .gdbcommands --args $(GDB_ARGS) \
			$(abs_builddir)/gtest $(GTEST_ARGS)

#
# Run unit tests with valgrind
#
test_valgrind: ucx gtest
	env LD_LIBRARY_PATH="$(VALGRIND_LIBPATH):${LD_LIBRARY_PATH}" \
	valgrind $(VALGRIND_ARGS) $(abs_builddir)/gtest $(GTEST_ARGS)
