#
# Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# See file LICENSE for terms.
#

if HAVE_CUDA

module_LTLIBRARIES      = libucs_nvtx.la
libucs_nvtx_la_CPPFLAGS = $(BASE_CPPFLAGS) $(CUDA_CPPFLAGS)
libucs_nvtx_la_CFLAGS   = $(BASE_CFLAGS) $(CUDA_CFLAGS)
libucs_nvtx_la_LIBADD   = $(CUDA_LIBS) \
                          $(top_builddir)/src/ucs/libucs.la
libucs_nvtx_la_LDFLAGS  = -version-info $(SOVERSION)
libucs_nvtx_la_SOURCES  = nvtx.c

endif
