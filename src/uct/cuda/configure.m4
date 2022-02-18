#
# Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_CUDA

AS_IF([test "x$cuda_happy" = "xyes"], [uct_modules="${uct_modules}:cuda"])
uct_cuda_modules=""
m4_include([src/uct/cuda/gdr_copy/configure.m4])
AC_DEFINE_UNQUOTED([uct_cuda_MODULES], ["${uct_cuda_modules}"], [CUDA loadable modules])
AC_CONFIG_FILES([src/uct/cuda/Makefile
                 src/uct/cuda/ucx-cuda.pc])
