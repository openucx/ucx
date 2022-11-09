#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

module_LTLIBRARIES            = libucs_topo_sysfs.la
libucs_topo_sysfs_la_CPPFLAGS = $(BASE_CPPFLAGS)
libucs_topo_sysfs_la_CFLAGS   = $(BASE_CFLAGS)
libucs_topo_sysfs_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la
libucs_topo_sysfs_la_LDFLAGS  = $(LDFLAGS) -version-info $(SOVERSION)

libucs_topo_sysfs_la_SOURCES  = \
    provider.c

PKG_CONFIG_NAME=topo-sysfs

include $(top_srcdir)/config/module.am
include $(top_srcdir)/config/module-pkg-config.am
