#
# Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_ROCM
if HAVE_GDR_COPY

module_LTLIBRARIES          = libuct_rocm_gdr.la
libuct_rocm_gdr_la_CPPFLAGS = $(BASE_CPPFLAGS) $(ROCM_CPPFLAGS) $(GDR_COPY_CPPFLAGS)
libuct_rocm_gdr_la_CFLAGS   = $(BASE_CFLAGS)
libuct_rocm_gdr_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                              $(top_builddir)/src/uct/rocm/libuct_rocm.la
libuct_rocm_gdr_la_LDFLAGS  = $(ROCM_LDFLAGS) $(GDR_COPY_LDFLAGS) -version-info $(SOVERSION)

noinst_HEADERS = \
	rocm_gdr_md.h \
	rocm_gdr_iface.h \
	rocm_gdr_ep.h

libuct_rocm_gdr_la_SOURCES = \
	rocm_gdr_md.c \
	rocm_gdr_iface.c \
	rocm_gdr_ep.c

PKG_CONFIG_NAME=rocm-gdr

include $(top_srcdir)/config/module.am
# TODO: enable pkg-config processing when module static build is enabled
# include $(top_srcdir)/config/module-pkg-config.am

endif
endif
