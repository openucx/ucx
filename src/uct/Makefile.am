#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
# $COPYRIGHT$
# $HEADER$
#

lib_LTLIBRARIES     = libuct.la

libuct_la_CFLAGS   =
libuct_la_LIBS     =
libuct_la_CPPFLAGS = -I$(abs_top_srcdir)/src -I$(abs_top_builddir)/src 
libuct_la_LDFLAGS  = -ldl -version-info $(SOVERSION)
libuct_la_LIBADD   = $(LIBM) ../ucs/libucs.la
libuct_ladir       = $(includedir)/uct

nobase_dist_libuct_la_HEADERS = \
	api/addr.h \
	api/tl.h \
	api/uct_def.h \
	api/uct.h \
	api/version.h

noinst_HEADERS = \
	tl/context.h \
	tl/tl_base.h \
	tl/tl_log.h

libuct_la_SOURCES = \
	tl/context.c \
	tl/memory.c \
	tl/tl_base.c

if HAVE_IB
libuct_la_CPPFLAGS += $(IBVERBS_CPPFLAGS)
libuct_la_LDFLAGS +=  $(IBVERBS_LDFLAGS)
noinst_HEADERS += \
	ib/base/ib_device.h \
	ib/base/ib_iface.h \
	ib/base/ib_log.h \
	ib/base/ib_verbs.h

libuct_la_SOURCES += \
	ib/base/ib_device.c \
	ib/base/ib_iface.c \
	ib/base/ib_log.c

if HAVE_MLX5_HW
libuct_la_SOURCES += \
	ib/mlx5/ib_mlx5_log.c \
	ib/mlx5/ib_mlx5.c
	
noinst_HEADERS += \
	ib/mlx5/ib_mlx5_log.h \
	ib/mlx5/ib_mlx5.h
endif


if HAVE_TL_RC
noinst_HEADERS += \
	ib/rc/base/rc_def.h \
	ib/rc/base/rc_ep.h \
	ib/rc/base/rc_iface.h \
	ib/rc/verbs/rc_verbs.h

libuct_la_SOURCES += \
	ib/rc/base/rc_ep.c \
	ib/rc/base/rc_iface.c \
	ib/rc/verbs/rc_verbs_ep.c \
	ib/rc/verbs/rc_verbs_iface.c

if HAVE_MLX5_HW
noinst_HEADERS += \
	ib/rc/accel/rc_mlx5.h

libuct_la_SOURCES += \
	ib/rc/accel/rc_mlx5_ep.c \
	ib/rc/accel/rc_mlx5_iface.c
endif
endif

if HAVE_TL_UD
libuct_la_SOURCES += \
	ib/ud/base/ud_iface.c \
	ib/ud/base/ud_ep.c \
	ib/ud/base/ud_log.c \
	ib/ud/verbs/ud_verbs.c

noinst_HEADERS += \
	ib/ud/base/ud_iface.h \
	ib/ud/base/ud_ep.h \
	ib/ud/base/ud_def.h \
	ib/ud/base/ud_inl.h \
	ib/ud/base/ud_log.h \
	ib/ud/verbs/ud_verbs.h 

if HAVE_MLX5_HW
noinst_HEADERS += \
	ib/ud/accel/ud_mlx5.h

libuct_la_SOURCES += \
	ib/ud/accel/ud_mlx5.c 
endif
endif

if HAVE_TL_CM
libuct_la_SOURCES += \
	ib/cm/cm_iface.c \
	ib/cm/cm_ep.c

noinst_HEADERS += \
	ib/cm/cm.h
endif

endif

if HAVE_CRAY_UGNI 
libuct_la_CFLAGS   += $(CRAY_UGNI_CFLAGS)
libuct_la_LDFLAGS  += $(CRAY_UGNI_LIBS)
libuct_la_LIBS     += $(CRAY_UGNI_LIBS)

noinst_HEADERS += \
    ugni/base/ugni_pd.h \
    ugni/base/ugni_device.h \
    ugni/base/ugni_iface.h \
    ugni/base/ugni_ep.h \
    ugni/rdma/ugni_rdma_iface.h \
    ugni/rdma/ugni_rdma_ep.h

libuct_la_SOURCES += \
    ugni/base/ugni_pd.c \
    ugni/base/ugni_device.c \
    ugni/base/ugni_iface.c \
    ugni/rdma/ugni_rdma_iface.c \
    ugni/rdma/ugni_rdma_ep.c
endif

if HAVE_CUDA
noinst_HEADERS += \
    cuda/cuda_pd.h \
    cuda/cuda_iface.h \
    cuda/cuda_ep.h

libuct_la_SOURCES += \
    cuda/cuda_pd.c \
    cuda/cuda_iface.c \
    cuda/cuda_ep.c
endif

# FIXME these should be done conditionally
noinst_HEADERS += \
    sm/mm/mm_iface.h \
    sm/mm/mm_ep.h \
    sm/mm/mm_def.h \
    sm/mm/mm_pd.h

libuct_la_SOURCES += \
    sm/mm/mm_iface.c \
    sm/mm/mm_ep.c \
    sm/mm/mm_pd.c \
    sm/mm/mm_sysv.c

# SGI / Cray XPMEM
if HAVE_XPMEM
libuct_la_CPPFLAGS += $(XPMEM_CPPFLAGS)
libuct_la_LDFLAGS +=  $(XPMEM_LDFLAGS)

libuct_la_SOURCES += \
    sm/mm/mm_xpmem.c
endif

# Cross Memory Attached
if HAVE_CMA
noinst_HEADERS += \
	sm/cma/cma_iface.h \
	sm/cma/cma_ep.h \
	sm/cma/cma_pd.h

libuct_la_SOURCES += \
	sm/cma/cma_iface.c \
	sm/cma/cma_ep.c \
	sm/cma/cma_pd.c
endif

# Inria's KNEM 
if HAVE_KNEM
noinst_HEADERS += \
	sm/knem/knem_iface.h \
	sm/knem/knem_ep.h \
	sm/knem/knem_pd.h

libuct_la_SOURCES += \
	sm/knem/knem_iface.c \
	sm/knem/knem_ep.c \
	sm/knem/knem_pd.c
endif
