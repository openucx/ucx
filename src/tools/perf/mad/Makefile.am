#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if HAVE_MAD

module_LTLIBRARIES              = libucx_perftest_mad.la
libucx_perftest_mad_la_CPPFLAGS = $(BASE_CPPFLAGS)
libucx_perftest_mad_la_CFLAGS   = $(BASE_CFLAGS) $(MAD_CFLAGS) \
                                  $(OPENMP_CFLAGS) $(LT_CFLAGS)
libucx_perftest_mad_la_LDFLAGS  = $(MAD_LDFLAGS) -version-info $(SOVERSION)
libucx_perftest_mad_la_LIBADD   = $(MAD_LIBS)
libucx_perftest_mad_la_SOURCES  = perftest_mad.c

include $(top_srcdir)/config/module.am

endif
