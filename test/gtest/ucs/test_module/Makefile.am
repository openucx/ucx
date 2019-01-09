#
# Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

module_LTLIBRARIES         = libtest_module.la
libtest_module_la_CPPFLAGS = $(BASE_CPPFLAGS)
libtest_module_la_CFLAGS   = $(BASE_CFLAGS)
libtest_module_la_LDFLAGS  = -version-info $(SOVERSION)
libtest_module_la_SOURCES  = test_module.c

include $(top_srcdir)/config/module.am

# disable install
install:
