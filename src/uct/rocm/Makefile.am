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
	cma/rocm_cma_md.h \
	cma/rocm_cma_iface.h \
	cma/rocm_cma_ep.h \
	cma/rocm_common.h

libuct_rocm_la_SOURCES = \
	cma/rocm_cma_md.c \
	cma/rocm_cma_iface.c \
	cma/rocm_cma_ep.c \
	cma/rocm_common.c

include $(top_srcdir)/config/module.am

endif
