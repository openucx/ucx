#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_ARG_WITH([doca-gpunetio],
            [AS_HELP_STRING([--with-doca-gpunetio=(DIR)],
                            [Use DOCA gpunetio (Default is guess)])],
            [with_doca_gpunetio=$withval],
            [with_doca_gpunetio=guess])

UCX_CHECK_CUDA

AS_IF([test "x$cuda_happy" = "xyes"],
      [
       # Default value
       GPUNETIO_CFLAGS=""
       AS_IF([test "x$with_doca_gpunetio" != "xno"],
             [
              AS_IF([test "x$with_doca_gpunetio" = "xguess"],
                    [
                     AS_IF([$PKG_CONFIG --exists doca-gpunetio],
                           [GPUNETIO_CFLAGS=$(pkg-config --cflags doca-gpunetio)])
                    ],
                    [
                     GPUNETIO_CFLAGS="-I${with_doca_gpunetio}/include"
                    ]) # "x$with_doca_gpunetio" != "xguess"
             ]) # "x$with_doca_gpunetio" != "xno"

       save_CPPFLAGS="$CPPFLAGS"
       CPPFLAGS="$CPPFLAGS $CUDA_CFLAGS $GPUNETIO_CFLAGS"

       gpunetio_happy=yes
       AC_CHECK_HEADERS([doca_gpunetio.h], [], [gpunetio_happy=no])

       CPPFLAGS="$save_CPPFLAGS"
       LDFLAGS="$save_LDFLAGS"
      ],
      [gpunetio_happy=no])

AS_IF([test "x$gpunetio_happy" = "xyes"],
      [
       uct_ib_mlx5_modules="${uct_ib_mlx5_modules}:gda"
       AC_SUBST(GPUNETIO_CFLAGS)
      ],
      [
       # gpunetio was requested but not found
       AS_IF([test "x$with_doca_gpunetio" != "xno" -a "x$with_doca_gpunetio" != "xguess"],
             [AC_MSG_ERROR([doca_gpunetio not found (cuda found: $cuda_happy)])])
      ])

AM_CONDITIONAL([HAVE_GPUNETIO], [test x$gpunetio_happy = xyes])
AC_CONFIG_FILES([src/uct/ib/mlx5/gdaki/Makefile
                 src/uct/ib/mlx5/gdaki/ucx-ib-mlx5-gda.pc])
