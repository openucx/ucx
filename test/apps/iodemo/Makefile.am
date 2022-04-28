#
# Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

bin_PROGRAMS = io_demo

noinst_HEADERS = \
	ucx_wrapper.h

io_demo_LDFLAGS  = -ldl

if IODEMO_CUDA
io_demo_CUDA_LIBS     = $(CUDA_LIBS)
io_demo_LDFLAGS      += $(CUDA_LDFLAGS)
io_demo_CUDA_CPPFLAGS = $(CUDA_CPPFLAGS) -DHAVE_CUDA
else
io_demo_CUDA_LIBS     =
io_demo_CUDA_CPPFLAGS =
endif


io_demo_CXXFLAGS = \
	$(BASE_CXXFLAGS)

io_demo_CPPFLAGS = $(BASE_CPPFLAGS) $(io_demo_CUDA_CPPFLAGS)

io_demo_LDADD    = \
	$(top_builddir)/src/ucm/libucm.la \
	$(top_builddir)/src/ucs/libucs.la \
	$(top_builddir)/src/uct/libuct.la \
	$(top_builddir)/src/ucp/libucp.la \
	$(io_demo_CUDA_LIBS)

io_demo_SOURCES = \
	ucx_wrapper.cc \
	io_demo.cc
