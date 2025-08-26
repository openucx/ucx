#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_GPUNETIO

module_LTLIBRARIES               = libuct_ib_mlx5_gdaki.la
libuct_ib_mlx5_gdaki_la_CPPFLAGS = $(BASE_CPPFLAGS) $(IBVERBS_CPPFLAGS) \
                                   $(CUDA_CPPFLAGS)
libuct_ib_mlx5_gdaki_la_CFLAGS   = $(BASE_CFLAGS) $(LT_CFLAGS) $(CUDA_CFLAGS) \
                                   $(GPUNETIO_CFLAGS)
libuct_ib_mlx5_gdaki_la_LDFLAGS  = $(IBVERBS_LDFLAGS) $(CUDA_LDFLAGS) \
                                   $(GPUNETIO_LDFLAGS) -version-info $(SOVERSION)
libuct_ib_mlx5_gdaki_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                                   $(top_builddir)/src/uct/libuct.la \
                                   $(top_builddir)/src/uct/ib/libuct_ib.la \
                                   $(top_builddir)/src/uct/ib/mlx5/libuct_ib_mlx5.la \
                                   $(top_builddir)/src/uct/cuda/libuct_cuda.la \
                                   $(CUDA_LIBS) $(GPUNETIO_LIBS)
libuct_ib_mlx5_gdaki_ladir       = $(includedir)/uct/ib_mlx5/gdaki

libuct_ib_mlx5_gdaki_la_SOURCES = \
	gdaki.c

noinst_HEADERS = \
	gdaki.h

PKG_CONFIG_NAME=mlx5-gdaki

include $(top_srcdir)/config/module.am
include $(top_srcdir)/config/module-pkg-config.am

endif
