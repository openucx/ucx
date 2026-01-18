#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_CUDA

save_NVCCFLAGS="$NVCCFLAGS"
NVCCFLAGS="$NVCCFLAGS -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT"
AC_MSG_CHECKING([checking cuda/atomic support])
AC_LANG_PUSH([CUDA])
AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
#include <cuda/atomic>
int v;
cuda::atomic_ref<int> ref{v};
]])],
[AC_MSG_RESULT([yes])
have_cuda_atomic_support=yes],
[AC_MSG_RESULT([no])])
AC_LANG_POP
NVCCFLAGS="$save_NVCCFLAGS"

AS_IF([test "x$cuda_happy" = "xyes"] && [test "x$have_mlx5" = "xyes"] &&
      ([test "$CUDA_MAJOR_VERSION" -eq 12 -a "$CUDA_MINOR_VERSION" -ge 2] ||
       [test "$CUDA_MAJOR_VERSION" -ge 13]) &&
      [test "x$have_mlx5dv_devx_umem" = "xyes"] &&
      [test "x$have_cuda_atomic_support" = "xyes"],
      [
       NVCCFLAGS="$NVCCFLAGS -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT"

       AS_IF([test "$CUDA_MAJOR_VERSION" -eq 12 -a "$CUDA_MINOR_VERSION" -ge 9] ||
             [test "$CUDA_MAJOR_VERSION" -ge 13],
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
