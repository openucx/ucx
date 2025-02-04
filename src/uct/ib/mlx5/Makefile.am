#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_MLX5_DV

module_LTLIBRARIES         = libuct_ib_mlx5.la
libuct_ib_mlx5_la_CPPFLAGS = $(BASE_CPPFLAGS) $(IBVERBS_CPPFLAGS)
libuct_ib_mlx5_la_CFLAGS   = $(BASE_CFLAGS) $(LT_CFLAGS)
libuct_ib_mlx5_la_LDFLAGS  = $(LIB_MLX5) $(IBVERBS_LDFLAGS) -version-info $(SOVERSION)
libuct_ib_mlx5_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                             $(top_builddir)/src/uct/libuct.la \
                             $(top_builddir)/src/uct/ib/libuct_ib.la

libuct_ib_mlx5_la_SOURCES = \
	ib_mlx5_log.c \
	ib_mlx5.c \
	dv/ib_mlx5_dv.c \
	dv/ib_mlx5dv_md.c

noinst_HEADERS = \
	ib_mlx5_log.h \
	ib_mlx5.h \
	ib_mlx5.inl \
	dv/ib_mlx5_dv.h \
	dv/ib_mlx5_ifc.h

if HAVE_TL_RC
noinst_HEADERS += \
	rc/rc_mlx5.h \
	rc/rc_mlx5.inl \
	rc/rc_mlx5_common.h

libuct_ib_mlx5_la_SOURCES += \
	rc/rc_mlx5_ep.c \
	rc/rc_mlx5_iface.c \
	rc/rc_mlx5_common.c

if HAVE_DEVX
libuct_ib_mlx5_la_SOURCES += \
	rc/rc_mlx5_devx.c

if HAVE_MLX5_MMO
libuct_ib_mlx5_la_SOURCES += \
	gga/gga_mlx5.c
endif # HAVE_MLX5_MMO
endif # HAVE_DEVX

endif # HAVE_TL_RC

if HAVE_TL_DC
noinst_HEADERS += \
	dc/dc_mlx5_ep.h \
	dc/dc_mlx5.inl \
	dc/dc_mlx5.h

libuct_ib_mlx5_la_SOURCES += \
	dc/dc_mlx5_ep.c \
	dc/dc_mlx5.c

if HAVE_DEVX
libuct_ib_mlx5_la_SOURCES += \
	dc/dc_mlx5_devx.c
endif # HAVE_DEVX

endif # HAVE_TL_DC

if HAVE_TL_UD
if HAVE_MLX5_HW_UD
noinst_HEADERS += \
	ud/ud_mlx5_common.h \
	ud/ud_mlx5.h

libuct_ib_mlx5_la_SOURCES += \
	ud/ud_mlx5_common.c \
	ud/ud_mlx5.c
endif # HAVE_MLX5_HW_UD

endif # HAVE_TL_UD

PKG_CONFIG_NAME=ib-mlx5

include $(top_srcdir)/config/module.am
include $(top_srcdir)/config/module-pkg-config.am

endif # HAVE_MLX5_DV
