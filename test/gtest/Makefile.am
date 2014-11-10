#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
#
# $COPYRIGHT$
# $HEADER$
#

UCS_HANDLE_ERRORS ?= freeze
export UCS_HANDLE_ERRORS

GTEST_FILTER ?= *
export GTEST_FILTER

VALGRIND_EXTRA_ARGS =

VALGRIND_ARGS = \
	--tool=memcheck \
	--leak-check=full \
	--track-origins=yes \
	--fair-sched=try

noinst_PROGRAMS = gtest

gtestdir  = $(includedir)
gtest_LDADD = \
	$(top_builddir)/src/ucs/libucs.la \
	$(top_builddir)/src/uct/libuct.la \
	$(top_builddir)/src/ucs/libucstest.la \
	$(top_builddir)/test/perf/libucxperf.la \
	$(GTEST_LIBS) \
	$(BOOST_THREAD_LIB)

gtest_CPPFLAGS = \
	-I$(top_srcdir)/src \
	-I$(top_srcdir)/test \
	-I$(top_builddir)/src \
	-I$(top_srcdir)/test/gtest \
	$(GTEST_CPPFLAGS)

gtest_LDFLAGS  = $(GTEST_LDFLAGS) -no-install
gtest_CXXFLAGS = $(GTEST_CXXFLAGS) -g -O3

gtest_SOURCES = \
	uct/test_uct_context.cc \
	uct/test_uct_perf.cc \
	ucs/test_component.cc \
	ucs/test_config.cc \
	ucs/test_datatype.cc \
	ucs/test_debug.cc \
	ucs/test_memtrack.cc \
	ucs/test_instr.cc \
	ucs/test_math.cc \
	ucs/test_mpool.cc \
	ucs/test_stats.cc \
	ucs/test_sys.cc \
	ucs/test_timer.cc \
	ucs/test_twheel.cc \
	ucs/test_frag_list.cc

.PHONY: test test gdb valgrind fix_rpath uct

all-local: gtest

uct:
	$(MAKE) -C $(top_builddir)/src/uct

test: uct gtest
	@rm -f core.*
	$(abs_builddir)/gtest --gtest_filter=$(GTEST_FILTER)
	
test_gdb: uct gtest
	echo 'r' > .gdbcommands
	env UCS_HANDLE_ERRORS=none UCS_GDB_PATH="" gdb -x .gdbcommands --args $(GDB_ARGS) $(abs_builddir)/gtest --gtest_filter=$(GTEST_FILTER)

list: gtest
	$(abs_builddir)/gtest --gtest_list_tests --gtest_filter=$(GTEST_FILTER)

test_valgrind: uct gtest
	env LD_LIBRARY_PATH="$(VALGRIND_LIBPATH):${LD_LIBRARY_PATH}" valgrind $(VALGRIND_ARGS) $(VALGRIND_EXTRA_ARGS) $(abs_builddir)/gtest --gtest_filter=$(GTEST_FILTER)
