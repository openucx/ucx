#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_CUDA

AS_IF([test "x$NVCC" != "x"] &&
      [test "x$have_mlx5dv_devx_umem" = "xyes"],
      [gda_happy=yes
       uct_ib_mlx5_modules="${uct_ib_mlx5_modules}:gda"],
      [gda_happy=no])

AM_CONDITIONAL([HAVE_GDA], [test x$gda_happy = xyes])
AC_CONFIG_FILES([src/uct/ib/mlx5/gdaki/Makefile
                 src/uct/ib/mlx5/gdaki/ucx-ib-mlx5-gda.pc])
