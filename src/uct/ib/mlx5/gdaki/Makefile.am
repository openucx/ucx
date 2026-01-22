#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_GPUNETIO

module_LTLIBRARIES             = libuct_ib_mlx5_gda.la
libuct_ib_mlx5_gda_la_CPPFLAGS = $(BASE_CPPFLAGS) $(IBVERBS_CPPFLAGS) \
                                 $(CUDA_CPPFLAGS)
libuct_ib_mlx5_gda_la_CFLAGS   = $(BASE_CFLAGS) $(LT_CFLAGS) $(CUDA_CFLAGS) \
                                 $(GPUNETIO_CFLAGS)
libuct_ib_mlx5_gda_la_LDFLAGS  = $(IBVERBS_LDFLAGS) $(CUDA_LDFLAGS) \
                                 -version-info $(SOVERSION)
libuct_ib_mlx5_gda_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                                 $(top_builddir)/src/uct/libuct.la \
                                 $(top_builddir)/src/uct/ib/libuct_ib.la \
                                 $(top_builddir)/src/uct/ib/mlx5/libuct_ib_mlx5.la \
                                 $(top_builddir)/src/uct/cuda/libuct_cuda.la \
                                 $(CUDA_LIBS)
libuct_ib_mlx5_gda_ladir       = $(includedir)/uct/ib/mlx5/gdaki

libuct_ib_mlx5_gda_la_SOURCES = \
	gdaki.c

nobase_dist_libuct_ib_mlx5_gda_la_HEADERS = \
	gdaki.cuh \
	gdaki_dev.h

noinst_HEADERS = \
	gdaki.h

PKG_CONFIG_NAME=ib-mlx5-gda

include $(top_srcdir)/config/module.am
include $(top_srcdir)/config/module-pkg-config.am

endif
