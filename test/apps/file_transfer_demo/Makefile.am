# Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# See file LICENSE for terms.

bin_PROGRAMS = file_transfer_demo

file_transfer_demo_CXXFLAGS = \
	-std=c++11 -I$(top_srcdir)/src

file_transfer_demo_LDADD_CXXFLAGS = \
	$(BASE_CXXFLAGS)

file_transfer_demo_LDADD_CPPFLAGS = $(BASE_CPPFLAGS)

file_transfer_demo_LDFLAGS = -ldl

file_transfer_demo_LDADD    = \
	$(top_builddir)/src/ucs/libucs.la \
	$(top_builddir)/src/uct/libuct.la \
	$(top_builddir)/src/ucp/libucp.la

file_transfer_demo_SOURCES = \
	file_transfer_demo.cc
