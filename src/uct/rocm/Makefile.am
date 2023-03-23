#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_ROCM

module_LTLIBRARIES      = libuct_rocm.la
libuct_rocm_la_CPPFLAGS = $(BASE_CPPFLAGS) $(ROCM_CPPFLAGS)
libuct_rocm_la_CFLAGS   = $(BASE_CFLAGS)
libuct_rocm_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                          $(top_builddir)/src/uct/libuct.la
libuct_rocm_la_LDFLAGS  = $(ROCM_LDFLAGS) $(ROCM_LIBS) -version-info $(SOVERSION) \
                          $(patsubst %, -Xlinker %, -L$(ROCM_ROOT)/lib -rpath $(ROCM_ROOT)/hip/lib -rpath $(ROCM_ROOT)/lib) \
                          $(patsubst %, -Xlinker %, --enable-new-dtags) \
                          $(patsubst %, -Xlinker %, -rpath $(ROCM_ROOT)/lib64)

noinst_HEADERS = \
	base/rocm_base.h \
	base/rocm_signal.h

libuct_rocm_la_SOURCES = \
	base/rocm_base.c \
	base/rocm_signal.c

noinst_HEADERS += \
    copy/rocm_copy_md.h \
    copy/rocm_copy_iface.h \
    copy/rocm_copy_cache.h \
    copy/rocm_copy_ep.h

libuct_rocm_la_SOURCES += \
    copy/rocm_copy_md.c \
    copy/rocm_copy_iface.c \
    copy/rocm_copy_cache.c \
    copy/rocm_copy_ep.c

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

PKG_CONFIG_NAME=rocm

include $(top_srcdir)/config/module.am
# TODO: enable pkg-config processing when module static build is enabled
# include $(top_srcdir)/config/module-pkg-config.am

endif
