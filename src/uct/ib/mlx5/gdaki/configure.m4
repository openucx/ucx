#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_CUDA

have_cuda_12_2=yes
AS_VERSION_COMPARE([$CUDA_VERSION], [12.2], [], [have_cuda_12_2=no])
AS_IF([test "x$cuda_happy" = "xyes"] && [test "x$have_mlx5" = "xyes"] &&
      [test "x$have_cuda_12_2" = "xyes"] &&
      [test "x$have_mlx5dv_devx_umem" = "xyes"] &&
      [test "x$have_cuda_atomic_support" = "xyes"],
      [
       AS_IF([test "x$NVCC_CXX_DIALECT" != "xc++17"],
             [
              NVCCFLAGS="$NVCCFLAGS -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT"
             ])

       have_cuda_12_9=yes
       AS_VERSION_COMPARE([$CUDA_VERSION], [12.9], [], [have_cuda_12_9=no])
       AS_IF([test "x$have_cuda_12_9" = "xyes"],
             [
              NVCCFLAGS="$NVCCFLAGS -D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE"
             ])

       # TODO check submodule version

       gda_happy=yes
      ],
      [gda_happy=no])
AS_IF([test "x$gda_happy" = "xyes"],
      [
       uct_ib_mlx5_modules="${uct_ib_mlx5_modules}:gda"
      ])

AM_CONDITIONAL([HAVE_GDA], [test x$gda_happy = xyes])
AC_CONFIG_FILES([src/uct/ib/mlx5/gdaki/Makefile
                 src/uct/ib/mlx5/gdaki/ucx-ib-mlx5-gda.pc])

