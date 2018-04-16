#
# Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

noinst_PROGRAMS = sa

noinst_HEADERS = \
	sa_base.h \
	sa_tcp.h \
	sa_ucx.h \
	sa_util.h

sa_CXXFLAGS = \
	-std=c++11 -g -Wall -Werror

sa_CPPFLAGS = \
	$(BASE_CPPFLAGS) \
	-I$(top_srcdir)/src \
	-I$(top_builddir)/src

sa_LDADD = \
	$(top_builddir)/src/ucs/libucs.la \
	$(top_builddir)/src/ucp/libucp.la

sa_SOURCES = \
	sa_base.cc \
	sa_main.cc \
	sa_tcp.cc \
	sa_ucx.cc \
	sa_util.cc

