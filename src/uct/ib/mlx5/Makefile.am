#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_MLX5

module_LTLIBRARIES         = libuct_ib_mlx5.la
libuct_ib_mlx5_la_CPPFLAGS = $(BASE_CPPFLAGS) $(IBVERBS_CPPFLAGS) $(MLX5_CPPFLAGS)
libuct_ib_mlx5_la_CFLAGS   = $(BASE_CFLAGS)
libuct_ib_mlx5_la_LDFLAGS  = $(IBVERBS_LDFLAGS) $(MLX5_LDFLAGS) -version-info $(SOVERSION)
libuct_ib_mlx5_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                             $(top_builddir)/src/uct/ib/libuct_ib.la \
                             $(IBVERBS_LIBS)

libmlx5_ver = $(shell (rpm -qf $(IBVERBS_DIR)/include/infiniband/mlx5_hw.h &>/dev/null && rpm -qf /usr/include/infiniband/mlx5_hw.h) | grep -v 'not owned' | head -1)

noinst_HEADERS =
libuct_ib_mlx5_la_SOURCES =

if HAVE_MLX5_HW
libuct_ib_mlx5_la_CPPFLAGS += -DUCT_IB_LIBMLX5_VER=\"$(libmlx5_ver)\"

noinst_HEADERS += \
	ib_mlx5_log.h \
	ib_mlx5.h \
	ib_mlx5.inl \
	dv/ib_mlx5_dv.h \
	dv/ib_mlx5_ifc.h \
	exp/ib_mlx5_hw.h

libuct_ib_mlx5_la_SOURCES += \
	ib_mlx5_log.c \
	ib_mlx5.c

if HAVE_EXP
noinst_HEADERS += \
	exp/ib_exp.h

libuct_ib_mlx5_la_SOURCES += \
	exp/ib_mlx5_hw.c \
	exp/ib_exp.c \
	exp/ib_exp_md.c
endif # HAVE_EXP

if HAVE_MLX5_DV
libuct_ib_mlx5_la_LIBADD += $(MLX5_LIBS)
libuct_ib_mlx5_la_SOURCES += \
	dv/ib_mlx5_dv.c \
	dv/ib_mlx5dv_md.c
endif # HAVE_MLX5_DV

endif # HAVE_MLX5_HW


if HAVE_TL_RC
if HAVE_MLX5_HW
noinst_HEADERS += \
	../rc/accel/rc_mlx5.h \
	../rc/accel/rc_mlx5.inl \
	../rc/accel/rc_mlx5_common.h

libuct_ib_mlx5_la_SOURCES += \
	../rc/accel/rc_mlx5_ep.c \
	../rc/accel/rc_mlx5_iface.c \
	../rc/accel/rc_mlx5_common.c
endif # HAVE_MLX5_HW

if HAVE_DEVX
libuct_ib_mlx5_la_SOURCES += \
	../rc/accel/rc_mlx5_devx.c
endif # HAVE_DEVX

endif # HAVE_TL_RC


if HAVE_TL_DC
noinst_HEADERS += \
	../dc/dc_mlx5_ep.h \
	../dc/dc_mlx5.inl \
	../dc/dc_mlx5.h

libuct_ib_mlx5_la_SOURCES += \
	../dc/dc_mlx5_ep.c \
	../dc/dc_mlx5.c

if HAVE_DEVX
libuct_ib_mlx5_la_SOURCES += \
	../dc/dc_mlx5_devx.c
endif # HAVE_DEVX

endif # HAVE_TL_DC


if HAVE_TL_UD
if HAVE_MLX5_HW_UD
noinst_HEADERS += \
	../ud/accel/ud_mlx5_common.h \
	../ud/accel/ud_mlx5.h

libuct_ib_mlx5_la_SOURCES += \
	../ud/accel/ud_mlx5_common.c \
	../ud/accel/ud_mlx5.c
endif # HAVE_MLX5_HW_UD

endif # HAVE_TL_UD

include $(top_srcdir)/config/module.am

endif # HAVE_MLX5
