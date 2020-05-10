#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#


noinst_lib_LTLIBRARIES = \
	libdlopen_test_do_mmap.la \
	libdlopen_test_do_load.la \
	libdlopen_test_do_load_rpath.la

libdlopen_test_do_mmap_la_SOURCES       = dlopen_test_do_mmap.c
libdlopen_test_do_load_la_SOURCES       = dlopen_test_do_load.c
libdlopen_test_do_load_rpath_la_SOURCES = dlopen_test_do_load.c
noinst_libdir = ${PWD}/.noinst

libdlopen_test_do_load_rpath_la_CPPFLAGS = -I$(top_srcdir)/src
libdlopen_test_do_load_la_CPPFLAGS       = -I$(top_srcdir)/src
libdlopen_test_do_load_rpath_la_LDFLAGS  = -R=${PWD}/rpath-subdir/.libs

SUBDIRS = rpath-subdir
