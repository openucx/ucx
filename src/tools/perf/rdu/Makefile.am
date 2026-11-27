#
# Copyright (C) SambaNova Systems, 2026. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if HAVE_RDU

module_LTLIBRARIES              = libucx_perftest_rdu.la
libucx_perftest_rdu_la_CPPFLAGS = $(BASE_CPPFLAGS)
libucx_perftest_rdu_la_CFLAGS   = $(BASE_CFLAGS) $(LT_CFLAGS)
libucx_perftest_rdu_la_LIBADD   = $(top_builddir)/src/uct/rdu/libuct_rdu.la
libucx_perftest_rdu_la_LDFLAGS  = -version-info $(SOVERSION) $(UCX_LT_RELEASE)
libucx_perftest_rdu_la_SOURCES  = rdu_alloc.c

include $(top_srcdir)/config/module.am

endif
