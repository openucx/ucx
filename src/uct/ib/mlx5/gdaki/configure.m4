#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_CUDA

AS_IF([test "x$cuda_happy" = "xyes"] && [test "x$have_mlx5" = "xyes"],
      [
       AS_IF([test "$CUDA_MAJOR_VERSION" -eq 12 -a "$CUDA_MINOR_VERSION" -ge 9] ||
             [test "$CUDA_MAJOR_VERSION" -ge 13],
             [
              GPUNETIO_CFLAGS="$GPUNETIO_CFLAGS -D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE"
             ])

       # TODO check submodule version

       gpunetio_happy=yes
      ],
      [gpunetio_happy=no])

AS_IF([test "x$gpunetio_happy" = "xyes"],
      [
       uct_ib_mlx5_modules="${uct_ib_mlx5_modules}:gda"
       AC_SUBST(GPUNETIO_CFLAGS)
      ])

AM_CONDITIONAL([HAVE_GPUNETIO], [test x$gpunetio_happy = xyes])
AC_CONFIG_FILES([src/uct/ib/mlx5/gdaki/Makefile
                 src/uct/ib/mlx5/gdaki/ucx-ib-mlx5-gda.pc])
