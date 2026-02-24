#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_IB

# Install plugin header (API definition)
libucx_plugin_ib_ladir = $(includedir)/uct/ib/plugin
nobase_dist_libucx_plugin_ib_la_HEADERS = \
	uct_ib_plugin.h

PKG_CONFIG_NAME=ib-plugin

include $(top_srcdir)/config/module-pkg-config.am

endif # HAVE_IB
