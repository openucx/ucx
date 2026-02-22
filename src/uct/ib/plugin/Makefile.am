#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_IB

# Plugin is built separately and dynamically loaded at runtime.
# Stub implementation is always in libuct_ib.so (see src/uct/ib/Makefile.am)
# Plugin library (libucx_plugin_ib.so) should be installed separately.
# UCX will discover and load it automatically via the plugin infrastructure.

# Install plugin header (API definition)
libucx_plugin_ib_ladir = $(includedir)/uct/ib/plugin
nobase_dist_libucx_plugin_ib_la_HEADERS = \
	uct_ib_plugin.h

PKG_CONFIG_NAME=ib-plugin

include $(top_srcdir)/config/module-pkg-config.am

endif # HAVE_IB
