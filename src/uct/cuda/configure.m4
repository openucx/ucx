#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2017. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_CUDA

AC_CHECK_DECLS([SYS_pidfd_open, SYS_pidfd_getfd],
               [], [],
               [#include <sys/syscall.h>])

# Provide fallback syscall numbers for kernel headers older than 5.3 / 5.6.
# The numbers are identical across these 64-bit Linux architectures:
# x86_64, aarch64, ppc64/ppc64le, riscv64.
AS_CASE([$host_cpu],
        [x86_64|aarch64|ppc64|ppc64le|riscv64],
            [AS_IF([test "x$ac_cv_have_decl_SYS_pidfd_open" != "xyes"],
                   [AC_DEFINE([SYS_pidfd_open], [434],
                              [Fallback syscall number for pidfd_open])])
             AS_IF([test "x$ac_cv_have_decl_SYS_pidfd_getfd" != "xyes"],
                   [AC_DEFINE([SYS_pidfd_getfd], [438],
                              [Fallback syscall number for pidfd_getfd])])])

AS_IF([test "x$cuda_happy" = "xyes"], [uct_modules="${uct_modules}:cuda"])
uct_cuda_modules=""
m4_include([src/uct/cuda/gdr_copy/configure.m4])
AC_DEFINE_UNQUOTED([uct_cuda_MODULES], ["${uct_cuda_modules}"], [CUDA loadable modules])
AC_CONFIG_FILES([src/uct/cuda/Makefile
                 src/uct/cuda/ucx-cuda.pc])
