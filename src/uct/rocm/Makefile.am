#
# Copyright (C) Advanced Micro Devices, Inc. 2016 - 2019. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_ROCM

SUBDIRS = gdr

module_LTLIBRARIES      = libuct_rocm.la
libuct_rocm_la_CPPFLAGS = $(BASE_CPPFLAGS) $(ROCM_CPPFLAGS)
libuct_rocm_la_CFLAGS   = $(BASE_CFLAGS) $(ROCM_CFLAGS)
libuct_rocm_la_LDFLAGS  = $(ROCM_LDFLAGS) -version-info $(SOVERSION)

noinst_HEADERS = \
	base/rocm_base.h

libuct_rocm_la_SOURCES = \
	base/rocm_base.c

noinst_HEADERS += \
    ipc/rocm_ipc_md.h \
    ipc/rocm_ipc_iface.h \
    ipc/rocm_ipc_ep.h \
    ipc/rocm_ipc_cache.h

libuct_rocm_la_SOURCES += \
    ipc/rocm_ipc_md.c \
    ipc/rocm_ipc_iface.c \
    ipc/rocm_ipc_ep.c \
    ipc/rocm_ipc_cache.c

noinst_HEADERS += \
    copy/rocm_copy_md.h \
    copy/rocm_copy_iface.h \
    copy/rocm_copy_ep.h

libuct_rocm_la_SOURCES += \
    copy/rocm_copy_md.c \
    copy/rocm_copy_iface.c \
    copy/rocm_copy_ep.c

include $(top_srcdir)/config/module.am

endif
