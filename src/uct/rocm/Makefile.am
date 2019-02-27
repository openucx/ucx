#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_ROCM

module_LTLIBRARIES      = libuct_rocm.la
libuct_rocm_la_CPPFLAGS = $(BASE_CPPFLAGS) $(ROCM_CPPFLAGS)
libuct_rocm_la_CFLAGS   = $(BASE_CFLAGS)
libuct_rocm_la_LDFLAGS  = $(ROCM_LDFLAGS) $(ROCM_LIBS) -version-info $(SOVERSION)

noinst_HEADERS = \
	rocm_cma_md.h \
	rocm_cma_iface.h \
	rocm_cma_ep.h \
	rocm_common.h

libuct_rocm_la_SOURCES = \
	rocm_cma_md.c \
	rocm_cma_iface.c \
	rocm_cma_ep.c \
	rocm_common.c

include $(top_srcdir)/config/module.am

endif
