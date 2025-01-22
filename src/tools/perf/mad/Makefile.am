#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
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
