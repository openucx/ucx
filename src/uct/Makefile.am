#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
# Copyright (c) The University of Tennesse and the University of Tennessee
#               Research Foundation. 2016.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

SUBDIRS = base cuda ib sm tcp

lib_LTLIBRARIES    = libuct.la
libuct_la_CFLAGS   = $(BASE_CFLAGS)
libuct_la_CPPFLAGS = $(BASE_CPPFLAGS)
libuct_la_LDFLAGS  = -ldl -version-info $(SOVERSION)
libuct_ladir       = $(includedir)/uct
libuct_la_SOURCES  =
noinst_HEADERS     =
libuct_la_LIBADD   = \
	../ucs/libucs.la \
	base/libuct_base.la \
	sm/libuct_sm.la \
	tcp/libuct_tcp.la

nobase_dist_libuct_la_HEADERS = \
	api/tl.h \
	api/uct_def.h \
	api/uct.h \
	api/version.h

if HAVE_CRAY_UGNI 
libuct_la_CFLAGS   += $(CRAY_UGNI_CFLAGS)
libuct_la_LDFLAGS  += $(CRAY_UGNI_LIBS)

noinst_HEADERS += \
    ugni/base/ugni_def.h \
    ugni/base/ugni_types.h \
    ugni/base/ugni_md.h \
    ugni/base/ugni_device.h \
    ugni/base/ugni_iface.h \
    ugni/base/ugni_ep.h \
    ugni/rdma/ugni_rdma_iface.h \
    ugni/rdma/ugni_rdma_ep.h \
    ugni/udt/ugni_udt_iface.h \
    ugni/udt/ugni_udt_ep.h \
    ugni/smsg/ugni_smsg_iface.h \
    ugni/smsg/ugni_smsg_ep.h

libuct_la_SOURCES += \
    ugni/base/ugni_md.c \
    ugni/base/ugni_device.c \
    ugni/base/ugni_iface.c \
    ugni/base/ugni_ep.c \
    ugni/rdma/ugni_rdma_iface.c \
    ugni/rdma/ugni_rdma_ep.c \
    ugni/udt/ugni_udt_iface.c \
    ugni/udt/ugni_udt_ep.c \
    ugni/smsg/ugni_smsg_iface.c	\
    ugni/smsg/ugni_smsg_ep.c
endif

if HAVE_ROCM
libuct_la_CPPFLAGS += $(ROCM_CPPFLAGS)
libuct_la_LDFLAGS +=  $(ROCM_LDFLAGS) $(ROCM_LIBS)

noinst_HEADERS += \
    rocm/rocm_cma_md.h \
    rocm/rocm_cma_iface.h \
    rocm/rocm_cma_ep.h \
    rocm/rocm_common.h

libuct_la_SOURCES += \
    rocm/rocm_cma_md.c \
    rocm/rocm_cma_iface.c \
    rocm/rocm_cma_ep.c \
    rocm/rocm_common.c
endif
