#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2001-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (C) UChicago Argonne, LLC. 2019.  ALL RIGHTS RESERVED.
# SPDX-License-Identifier: BSD-3-Clause
# See file LICENSE for terms.
#

if HAVE_XPMEM

module_LTLIBRARIES       = libuct_xpmem.la
libuct_xpmem_la_CFLAGS   = $(BASE_CFLAGS) $(XPMEM_CFLAGS) $(LT_CFLAGS)
libuct_xpmem_la_CPPFLAGS = $(BASE_CPPFLAGS)
libuct_xpmem_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                           $(top_builddir)/src/uct/libuct.la
libuct_xpmem_la_LDFLAGS  = $(XPMEM_LIBS) -version-info $(SOVERSION)
libuct_xpmem_la_SOURCES  = mm_xpmem.c

PKG_CONFIG_NAME=xpmem

include $(top_srcdir)/config/module.am
include $(top_srcdir)/config/module-pkg-config.am

endif
