#
# Copyright (C) Intel Corporation, 2023. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if HAVE_ZE

module_LTLIBRARIES             = libucx_perftest_ze.la
libucx_perftest_ze_la_CPPFLAGS = $(BASE_CPPFLAGS) $(ZE_CPPFLAGS)
libucx_perftest_ze_la_CFLAGS   = $(BASE_CFLAGS) $(ZE_CFLAGS) \
                                 $(LT_CFLAGS)
libucx_perftest_ze_la_LDFLAGS  = $(ZE_LDFLAGS) $(ZE_LIBS) -version-info $(SOVERSION)
libucx_perftest_ze_la_SOURCES  = ze_alloc.c

include $(top_srcdir)/config/module.am

endif
