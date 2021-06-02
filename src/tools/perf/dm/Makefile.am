#
# Copyright (C) Huawei Technologies Co., Ltd. 2021.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if HAVE_IB_DM

module_LTLIBRARIES               = libucx_perftest_dm.la
libucx_perftest_dm_la_CPPFLAGS = $(BASE_CPPFLAGS)
libucx_perftest_dm_la_CFLAGS   = $(BASE_CFLAGS)
libucx_perftest_dm_la_LDFLAGS  = $(BASE_LDLAGS) -version-info $(SOVERSION)
libucx_perftest_dm_la_SOURCES  = dm_alloc.c

include $(top_srcdir)/config/module.am

endif
