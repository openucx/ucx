#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2001-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# See file LICENSE for terms.
#

module_LTLIBRARIES         = libtest_module.la
libtest_module_la_CPPFLAGS = $(BASE_CPPFLAGS)
libtest_module_la_CFLAGS   = $(BASE_CFLAGS) $(LT_CFLAGS)
libtest_module_la_LDFLAGS  = -version-info $(SOVERSION)
libtest_module_la_SOURCES  = test_module.c

include $(top_srcdir)/config/module.am

# disable install
install:
