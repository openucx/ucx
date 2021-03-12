#
# Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
# Copyright (c) NVIDIA Corporation. 2021. All rights reserved.
#
# See file LICENSE for terms.
#

if HAVE_NVTX

module_LTLIBRARIES      = libucs_nvtx.la
libucs_nvtx_la_CPPFLAGS = $(BASE_CPPFLAGS) $(NVTX_CPPFLAGS)
libucs_nvtx_la_CFLAGS   = $(BASE_CFLAGS) $(NVTX_CFLAGS)
libucs_nvtx_la_LIBADD   = $(NVTX_LIBS) \
                          $(top_builddir)/src/ucs/libucs.la
libucs_nvtx_la_LDFLAGS  = -version-info $(SOVERSION)
libucs_nvtx_la_SOURCES  = nvtx.c
include $(top_srcdir)/config/module.am

endif
