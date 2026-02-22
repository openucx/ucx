#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_IB

# Plugin library is only built if real implementation is available
# Stub implementation is always in libuct_ib.so (see src/uct/ib/Makefile.am)
# Library name follows new convention: libucx_plugin_<name>.so
if HAVE_IB_PLUGIN
module_LTLIBRARIES             = libucx_plugin_ib.la
libucx_plugin_ib_la_CPPFLAGS   = $(BASE_CPPFLAGS) $(IBVERBS_CPPFLAGS)
libucx_plugin_ib_la_CFLAGS     = $(BASE_CFLAGS) $(LT_CFLAGS)
libucx_plugin_ib_la_LDFLAGS    = $(IBVERBS_LDFLAGS) -version-info $(SOVERSION)
libucx_plugin_ib_la_LIBADD    = $(top_builddir)/src/ucs/libucs.la \
                                 $(top_builddir)/src/uct/libuct.la
# Note: Plugin accesses libuct_ib symbols at runtime, no link dependency needed
# to avoid circular dependency

libucx_plugin_ib_la_SOURCES = \
	ucx_plugin/src/ucx_plugin.c

# Install plugin header
libucx_plugin_ib_ladir = $(includedir)/uct/ib/plugin
nobase_dist_libucx_plugin_ib_la_HEADERS = \
	uct_ib_plugin.h

PKG_CONFIG_NAME=ib-plugin

include $(top_srcdir)/config/module.am
include $(top_srcdir)/config/module-pkg-config.am
endif # HAVE_IB_PLUGIN

endif # HAVE_IB