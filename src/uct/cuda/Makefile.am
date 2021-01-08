#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_CUDA

SUBDIRS = . gdr_copy

module_LTLIBRARIES      = libuct_cuda.la
libuct_cuda_la_CPPFLAGS = $(BASE_CPPFLAGS) $(CUDA_CPPFLAGS)
libuct_cuda_la_CFLAGS   = $(BASE_CFLAGS) $(CUDA_CFLAGS)
libuct_cuda_la_LDFLAGS  = $(CUDA_LDFLAGS) -version-info $(SOVERSION)
libuct_cuda_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                          $(top_builddir)/src/uct/libuct.la \
                          $(CUDA_LIBS)

noinst_HEADERS = \
	base/cuda_md.h \
	base/cuda_iface.h \
	cuda_copy/cuda_copy_md.h \
	cuda_copy/cuda_copy_iface.h \
	cuda_copy/cuda_copy_ep.h \
	cuda_ipc/cuda_ipc_md.h \
	cuda_ipc/cuda_ipc_iface.h \
	cuda_ipc/cuda_ipc_ep.h \
	cuda_ipc/cuda_ipc_cache.h

libuct_cuda_la_SOURCES = \
	base/cuda_iface.c \
	base/cuda_md.c \
	cuda_copy/cuda_copy_md.c \
	cuda_copy/cuda_copy_iface.c \
	cuda_copy/cuda_copy_ep.c \
	cuda_ipc/cuda_ipc_md.c \
	cuda_ipc/cuda_ipc_iface.c \
	cuda_ipc/cuda_ipc_ep.c \
	cuda_ipc/cuda_ipc_cache.c

include $(top_srcdir)/config/module.am

endif
