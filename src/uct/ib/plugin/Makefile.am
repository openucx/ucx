#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_IB

pluginincludedir      = $(includedir)/uct/ib/plugin
plugininclude_HEADERS = uct_ib_plugin.h

noinst_HEADERS = uct_ib_plugin.h

noinst_LTLIBRARIES                 = libuct_ib_plugin_stubs.la
libuct_ib_plugin_stubs_la_SOURCES  = uct_ib_plugin_stubs.c
libuct_ib_plugin_stubs_la_CPPFLAGS = $(BASE_CPPFLAGS)
libuct_ib_plugin_stubs_la_CFLAGS   = $(BASE_CFLAGS)

endif # HAVE_IB
