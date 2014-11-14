#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
#
# $COPYRIGHT$
# $HEADER$
#

lib_LTLIBRARIES     = libuct.la

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
	tl/context.c

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
noinst_HEADERS += \
	ib/mlx5/ib_mlx5.h

libuct_la_SOURCES += \
	ib/mlx5/ib_mlx5.c
endif


if HAVE_TL_RC
noinst_HEADERS += \
	ib/rc/rc_ep.h \
	ib/rc/rc_iface.h

libuct_la_SOURCES += \
	ib/rc/rc_ep.c \
	ib/rc/rc_iface.c

if HAVE_MLX5_HW
libuct_la_SOURCES += \
	ib/rc/rc_mlx5.c \
	ib/rc/rc_mlx5.h
endif

endif
endif
