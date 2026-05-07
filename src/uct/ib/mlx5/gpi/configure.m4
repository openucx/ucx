#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_CUDA

AC_ARG_WITH([gpi],
            [AS_HELP_STRING([--without-gpi], [Disable GPI (GPU Push Interface)])],
            [], [with_gpi=yes])

AS_IF([test "x$with_gpi" = "xyes"] &&
      [test "x$cuda_happy" = "xyes"] &&
      [gpi_happy=yes], [gpi_happy=no])

AS_IF([test "x$gpi_happy" = "xyes"],
      [uct_ib_mlx5_modules="${uct_ib_mlx5_modules}:gpi"])

AM_CONDITIONAL([HAVE_GPI], [test "x$gpi_happy" != "xno"])
AC_CONFIG_FILES([src/uct/ib/mlx5/gpi/Makefile
                 src/uct/ib/mlx5/gpi/ucx-ib-mlx5-gpi.pc])
