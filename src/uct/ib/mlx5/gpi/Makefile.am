#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_GPI

module_LTLIBRARIES             = libuct_ib_mlx5_gpi.la
libuct_ib_mlx5_gpi_la_CPPFLAGS = $(BASE_CPPFLAGS) $(IBVERBS_CPPFLAGS) \
                                 $(CUDA_CPPFLAGS)
libuct_ib_mlx5_gpi_la_CFLAGS   = $(BASE_CFLAGS) $(LT_CFLAGS) $(CUDA_CFLAGS)
libuct_ib_mlx5_gpi_la_LDFLAGS  = $(IBVERBS_LDFLAGS) $(CUDA_LDFLAGS) \
                                 -version-info $(SOVERSION)
libuct_ib_mlx5_gpi_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                                 $(top_builddir)/src/uct/libuct.la \
                                 $(top_builddir)/src/uct/ib/libuct_ib.la \
                                 $(top_builddir)/src/uct/ib/mlx5/libuct_ib_mlx5.la \
                                 $(top_builddir)/src/uct/cuda/libuct_cuda.la \
                                 $(CUDA_LIBS)
libuct_ib_mlx5_gpi_ladir       = $(includedir)/uct/ib/mlx5/gpi

libuct_ib_mlx5_gpi_la_SOURCES = \
	gpi.c

noinst_HEADERS = \
	gpi.h

PKG_CONFIG_NAME=ib-mlx5-gpi

include $(top_srcdir)/config/module.am
include $(top_srcdir)/config/module-pkg-config.am

endif
