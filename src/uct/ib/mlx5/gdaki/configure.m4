#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_CUDA

AC_ARG_WITH([gda],
            [AS_HELP_STRING([--without-gda], [Disable GDA-KI])],
            [], [with_gda=yes])

GDA_GPUNETIO_HEADER="$srcdir/src/uct/ib/mlx5/gdaki/gpunetio/device/doca_gpunetio_dev_verbs_qp.cuh"

AS_IF([test "x$with_gda" = "xyes"] && [test "x$cuda_happy" = "xyes"],
      [AS_IF([test -f "$GDA_GPUNETIO_HEADER"],
             [gda_happy=yes],
             [AC_MSG_WARN([GDA-KI is disabled: gpunetio headers not found at $GDA_GPUNETIO_HEADER. Run 'git submodule update --init external/gpunetio'.])
              gda_happy=no])],
      [gda_happy=no])

AS_IF([test "x$gda_happy" = "xyes"],
      [uct_ib_mlx5_modules="${uct_ib_mlx5_modules}:gda"
       AC_DEFINE([HAVE_GDA], [1], [Enable GDA-KI support])])

AM_CONDITIONAL([HAVE_GDA], [test "x$gda_happy" != "xno"])
AC_CONFIG_FILES([src/uct/ib/mlx5/gdaki/Makefile
                 src/uct/ib/mlx5/gdaki/ucx-ib-mlx5-gda.pc])
