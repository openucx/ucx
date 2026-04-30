#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_CUDA

AC_ARG_WITH([gda],
            [AS_HELP_STRING([--without-gda], [Disable GDA-KI])],
            [], [with_gda=yes])

AS_IF([test "x$with_gda" = "xyes"] && [test "x$cuda_happy" = "xyes"], [
    uct_ib_mlx5_modules="${uct_ib_mlx5_modules}:gda"], [with_gda=no])

AM_CONDITIONAL([HAVE_GDA], [test "x$with_gda" != "xno"])
AC_CONFIG_FILES([src/uct/ib/mlx5/gdaki/Makefile
                 src/uct/ib/mlx5/gdaki/ucx-ib-mlx5-gda.pc])
