#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_IB

SUBDIRS = . cm rdmacm

module_LTLIBRARIES    = libuct_ib.la
libuct_ib_la_CPPFLAGS = $(BASE_CPPFLAGS) $(IBVERBS_CPPFLAGS)
libuct_ib_la_CFLAGS   = $(BASE_CFLAGS)
libuct_ib_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                        $(top_builddir)/src/uct/libuct.la
libuct_ib_la_LDFLAGS  = $(IBVERBS_LDFLAGS) $(NUMA_LIBS) -version-info $(SOVERSION)
libmlx5_ver           = $(shell (rpm -qf $(IBVERBS_DIR)/include/infiniband/mlx5_hw.h &>/dev/null && rpm -qf /usr/include/infiniband/mlx5_hw.h) | grep -v 'not owned' | head -1)

noinst_HEADERS = \
	base/ib_device.h \
	base/ib_iface.h \
	base/ib_log.h \
	base/ib_md.h \
	base/ib_verbs.h

libuct_ib_la_SOURCES = \
	base/ib_device.c \
	base/ib_iface.c \
	base/ib_log.c \
	base/ib_md.c

# TODO separate module for mlx5
if HAVE_MLX5_HW
libuct_ib_la_CPPFLAGS += -DUCT_IB_LIBMLX5_VER=\"$(libmlx5_ver)\"

noinst_HEADERS += \
	mlx5/ib_mlx5_log.h \
	mlx5/ib_mlx5.h \
	mlx5/ib_mlx5.inl \
	mlx5/dv/ib_mlx5_dv.h \
	mlx5/dv/ib_mlx5_ifc.h \
	mlx5/exp/ib_mlx5_hw.h

libuct_ib_la_SOURCES += \
	mlx5/ib_mlx5_log.c \
	mlx5/ib_mlx5.c

if HAVE_EXP
noinst_HEADERS += \
	mlx5/exp/ib_exp.h

libuct_ib_la_SOURCES += \
	mlx5/exp/ib_mlx5_hw.c \
	mlx5/exp/ib_exp.c \
	mlx5/exp/ib_exp_md.c
endif # HAVE_EXP

if HAVE_MLX5_DV
libuct_ib_la_LDFLAGS +=  $(LIB_MLX5)
libuct_ib_la_SOURCES += \
	mlx5/dv/ib_mlx5_dv.c \
	mlx5/dv/ib_mlx5dv_md.c
endif # HAVE_MLX5_DV

endif # HAVE_MLX5_HW


if HAVE_TL_RC
noinst_HEADERS += \
	rc/base/rc_def.h \
	rc/base/rc_ep.h \
	rc/base/rc_iface.h \
	rc/verbs/rc_verbs.h \
	rc/verbs/rc_verbs_impl.h

libuct_ib_la_SOURCES += \
	rc/base/rc_ep.c \
	rc/base/rc_iface.c \
	rc/verbs/rc_verbs_ep.c \
	rc/verbs/rc_verbs_iface.c

if HAVE_MLX5_HW
noinst_HEADERS += \
	rc/accel/rc_mlx5.h \
	rc/accel/rc_mlx5.inl \
	rc/accel/rc_mlx5_common.h

libuct_ib_la_SOURCES += \
	rc/accel/rc_mlx5_ep.c \
	rc/accel/rc_mlx5_iface.c \
	rc/accel/rc_mlx5_common.c
endif # HAVE_MLX5_HW

if HAVE_DEVX
libuct_ib_la_SOURCES += \
	rc/accel/rc_mlx5_devx.c
endif # HAVE_DEVX

endif # HAVE_TL_RC


if HAVE_TL_DC
noinst_HEADERS += \
	dc/dc_mlx5_ep.h \
	dc/dc_mlx5.inl \
	dc/dc_mlx5.h

libuct_ib_la_SOURCES += \
	dc/dc_mlx5_ep.c \
	dc/dc_mlx5.c

if HAVE_DEVX
libuct_ib_la_SOURCES += \
	dc/dc_mlx5_devx.c
endif # HAVE_DEVX

endif # HAVE_TL_DC


if HAVE_TL_UD
noinst_HEADERS += \
	ud/base/ud_iface_common.h \
	ud/base/ud_iface.h \
	ud/base/ud_ep.h \
	ud/base/ud_def.h \
	ud/base/ud_inl.h \
	ud/verbs/ud_verbs.h

libuct_ib_la_SOURCES += \
	ud/base/ud_iface_common.c \
	ud/base/ud_iface.c \
	ud/base/ud_ep.c \
	ud/base/ud_log.c \
	ud/verbs/ud_verbs.c


if HAVE_MLX5_HW_UD
noinst_HEADERS += \
	ud/accel/ud_mlx5_common.h \
	ud/accel/ud_mlx5.h

libuct_ib_la_SOURCES += \
	ud/accel/ud_mlx5_common.c \
	ud/accel/ud_mlx5.c
endif # HAVE_MLX5_HW_UD

endif # HAVE_TL_UD

include $(top_srcdir)/config/module.am

endif # HAVE_IB
