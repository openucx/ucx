#
# Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

noinst_PROGRAMS = io_demo

noinst_HEADERS = \
	ucx_wrapper.h

io_demo_CXXFLAGS = \
	$(BASE_CXXFLAGS)

io_demo_CPPFLAGS = $(BASE_CPPFLAGS)

io_demo_LDADD = \
	$(top_builddir)/src/ucs/libucs.la \
	$(top_builddir)/src/ucp/libucp.la

io_demo_SOURCES = \
	ucx_wrapper.cc \
	io_demo.cc
