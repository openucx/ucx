#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
# $COPYRIGHT$
# $HEADER$
#

lib_LTLIBRARIES     = libuct.la

libuct_la_CFLAGS   =
libuct_la_LIBS     =
libuct_la_CPPFLAGS = -I$(abs_top_srcdir)/src -I$(abs_top_builddir)/src 
libuct_la_LDFLAGS  = -ldl -version-info $(SOVERSION)
libuct_la_LIBADD   = $(LIBM) ../ucs/libucs.la
libuct_ladir       = $(includedir)

nobase_dist_libuct_la_HEADERS = \
	api/tl.h \
	api/uct_def.h \
	api/uct.h \
	api/version.h

noinst_HEADERS = \
	tl/context.h

libuct_la_SOURCES = \
	tl/context.c \
	tl/tl.c

if HAVE_IB
libuct_la_CPPFLAGS += $(IBVERBS_CPPFLAGS)
libuct_la_LDFLAGS +=  $(IBVERBS_LDFLAGS)
noinst_HEADERS += \
	ib/base/ib_context.h \
	ib/base/ib_device.h \
	ib/base/ib_iface.h \
	ib/base/ib_verbs.h

libuct_la_SOURCES += \
	ib/base/ib_context.c \
	ib/base/ib_device.c \
	ib/base/ib_iface.c

if HAVE_MLX5_HW
libuct_la_SOURCES += \
	ib/mlx5/ib_mlx5.c
	
noinst_HEADERS += \
	ib/mlx5/ib_mlx5.h
endif


if HAVE_TL_RC
noinst_HEADERS += \
	ib/rc/rc_def.h \
	ib/rc/rc_ep.h \
	ib/rc/rc_iface.h

libuct_la_SOURCES += \
	ib/rc/rc_ep.c \
	ib/rc/rc_iface.c \
	ib/rc/rc_verbs.c

if HAVE_MLX5_HW
libuct_la_SOURCES += \
	ib/rc/rc_mlx5.c
endif

endif
endif

if HAVE_CRAY_UGNI 
libuct_la_CFLAGS   += $(CRAY_UGNI_CFLAGS)
libuct_la_LDFLAGS  += $(CRAY_UGNI_LIBS)
libuct_la_LIBS     += $(CRAY_UGNI_LIBS)

noinst_HEADERS += \
    ugni/ugni_context.h \
    ugni/ugni_device.h \
    ugni/ugni_iface.h

libuct_la_SOURCES += \
    ugni/ugni_context.c \
    ugni/ugni_device.c \
    ugni/ugni_iface.c
endif
