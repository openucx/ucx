#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# See file LICENSE for terms.
#

if HAVE_FUSE3

bin_PROGRAMS     = ucx_vfs
ucx_vfs_CPPFLAGS = $(BASE_CPPFLAGS) $(FUSE3_CPPFLAGS)
ucx_vfs_LDFLAGS  = $(FUSE3_LDFLAGS)
ucx_vfs_CFLAGS   = $(BASE_CFLAGS)
ucx_vfs_SOURCES  = vfs_main.c vfs_server.c
noinst_HEADERS   = vfs_daemon.h
ucx_vfs_LDADD    = $(FUSE3_LIBS) \
                   $(top_builddir)/src/ucs/vfs/sock/libucs_vfs_sock.la \
                   $(top_builddir)/src/ucs/libucs.la

endif
