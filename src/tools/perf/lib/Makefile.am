#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2015-2017. ALL RIGHTS RESERVED.
# Copyright (C) The University of Tennessee and The University
#               of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
# Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

noinst_LTLIBRARIES     = libucxperf.la
libucxperf_la_CPPFLAGS = $(BASE_CPPFLAGS) $(RTE_CPPFLAGS)
libucxperf_la_CFLAGS   = $(BASE_CFLAGS) $(OPENMP_CFLAGS)
libucxperf_la_CXXFLAGS = $(BASE_CXXFLAGS) $(OPENMP_CFLAGS)
libucxperf_la_LDFLAGS  = $(RTE_LDFLAGS) $(OPENMP_CFLAGS)
libucxperf_la_LIBADD   = \
	$(abs_top_builddir)/src/uct/libuct.la \
	$(abs_top_builddir)/src/ucp/libucp.la \
	$(abs_top_builddir)/src/ucs/libucs.la

# C-linkable C++ code - must override any inherited CXXFLAGS
CXXFLAGS              += -nostdlib $(PERF_LIB_CXXFLAGS) -std=c++11

noinst_HEADERS = \
	libperf_int.h

libucxperf_la_SOURCES = \
	libperf.c \
	libperf_memory.c \
	libperf_thread.c \
	uct_tests.cc \
	ucp_tests.cc
