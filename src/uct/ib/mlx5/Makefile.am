#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_MLX5_DV

module_LTLIBRARIES      = libuct_ib_mlx5.la
libuct_ib_mlx5_la_CPPFLAGS = $(BASE_CPPFLAGS) $(IBVERBS_CPPFLAGS)
libuct_ib_mlx5_la_CFLAGS   = $(BASE_CFLAGS)
libuct_ib_mlx5_la_LDFLAGS  = $(LIB_MLX5) $(IBVERBS_LDFLAGS) -version-info $(SOVERSION)
libuct_ib_mlx5_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                             $(top_builddir)/src/uct/libuct.la \
                             $(top_builddir)/src/uct/ib/libuct_ib.la
libuct_ib_mlx5_la_SOURCES =

noinst_HEADERS =

libmlx5_ver             = $(shell (rpm -qf $(IBVERBS_DIR)/include/infiniband/mlx5_hw.h &>/dev/null && rpm -qf /usr/include/infiniband/mlx5_hw.h) | grep -v 'not owned' | head -1)

libuct_ib_mlx5_la_CPPFLAGS += -DUCT_IB_LIBMLX5_VER=\"$(libmlx5_ver)\"

noinst_HEADERS += \
	ib_mlx5_log.h \
	ib_mlx5.h \
	ib_mlx5.inl \
	dv/ib_mlx5_dv.h \
	dv/ib_mlx5_ifc.h

libuct_ib_mlx5_la_SOURCES += \
	ib_mlx5_log.c \
	ib_mlx5.c

libuct_ib_mlx5_la_SOURCES += \
	dv/ib_mlx5_dv.c \
	dv/ib_mlx5dv_md.c

if HAVE_TL_RC
noinst_HEADERS += \
	accel/rc_mlx5.h \
	accel/rc_mlx5.inl \
	accel/rc_mlx5_common.h

libuct_ib_mlx5_la_SOURCES += \
	accel/rc_mlx5_ep.c \
	accel/rc_mlx5_iface.c \
	accel/rc_mlx5_common.c

if HAVE_DEVX
libuct_ib_mlx5_la_SOURCES += \
	accel/rc_mlx5_devx.c \
	accel/gga_mlx5.c

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
	accel/ud_mlx5_common.h \
	accel/ud_mlx5.h

libuct_ib_mlx5_la_SOURCES += \
	accel/ud_mlx5_common.c \
	accel/ud_mlx5.c
endif # HAVE_MLX5_HW_UD

endif # HAVE_TL_UD

PKG_CONFIG_NAME=ib-mlx5

include $(top_srcdir)/config/module.am
include $(top_srcdir)/config/module-pkg-config.am

endif # HAVE_MLX5_DV
