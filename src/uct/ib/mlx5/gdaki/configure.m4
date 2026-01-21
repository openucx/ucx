#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_CUDA

UCX_CHECK_CUDA_GE(12, 2)
AS_IF([test "x$cuda_happy" = "xyes"] && [test "x$have_mlx5" = "xyes"] &&
      [test "x$ucx_cuda_ge" = "xyes" ] &&
      [test "x$have_mlx5dv_devx_umem" = "xyes"] &&
      [test "x$have_cuda_atomic_support" = "xyes"],
      [
       AS_IF([test "x$NVCC_CXX_DIALECT" != "xc++17"],
             [
              NVCCFLAGS="$NVCCFLAGS -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT"
             ])

       UCX_CHECK_CUDA_GE(12, 9)
       AS_IF([test "x$ucx_cuda_ge" = "xyes" ],
             [
              NVCCFLAGS="$NVCCFLAGS -D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE"
             ])

       # TODO check submodule version

       gpunetio_happy=yes
      ],
      [gpunetio_happy=no])

AS_IF([test "x$gpunetio_happy" = "xyes"],
      [
       uct_ib_mlx5_modules="${uct_ib_mlx5_modules}:gda"
      ])

AM_CONDITIONAL([HAVE_GPUNETIO], [test x$gpunetio_happy = xyes])
AC_CONFIG_FILES([src/uct/ib/mlx5/gdaki/Makefile
                 src/uct/ib/mlx5/gdaki/ucx-ib-mlx5-gda.pc])

