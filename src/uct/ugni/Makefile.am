#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_CRAY_UGNI

module_LTLIBRARIES      = libuct_ugni.la
libuct_ugni_la_CPPFLAGS = $(BASE_CPPFLAGS)
libuct_ugni_la_CFLAGS   = $(BASE_CFLAGS) $(CRAY_UGNI_CFLAGS)
libuct_ugni_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                          $(top_builddir)/src/uct/libuct.la
libuct_ugni_la_LDFLAGS  = $(CRAY_UGNI_LIBS) -version-info $(SOVERSION)

noinst_HEADERS = \
	base/ugni_def.h \
	base/ugni_types.h \
	base/ugni_md.h \
	base/ugni_device.h \
	base/ugni_iface.h \
	base/ugni_ep.h \
	rdma/ugni_rdma_iface.h \
	rdma/ugni_rdma_ep.h \
	udt/ugni_udt_iface.h \
	udt/ugni_udt_ep.h \
	smsg/ugni_smsg_iface.h \
	smsg/ugni_smsg_ep.h

libuct_ugni_la_SOURCES = \
	base/ugni_md.c \
	base/ugni_device.c \
	base/ugni_iface.c \
	base/ugni_ep.c \
	rdma/ugni_rdma_iface.c \
	rdma/ugni_rdma_ep.c \
	udt/ugni_udt_iface.c \
	udt/ugni_udt_ep.c \
	smsg/ugni_smsg_iface.c	\
	smsg/ugni_smsg_ep.c

PKG_CONFIG_NAME=ugni

include $(top_srcdir)/config/module.am
# TODO: enable pkg-config processing when module static build is enabled
# include $(top_srcdir)/config/module-pkg-config.am

endif
