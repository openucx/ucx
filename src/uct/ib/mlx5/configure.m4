#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

#
# Add IB mlx5 provider support
#

AC_CACHE_CHECK([for AArch64 ST64B assembler support],
               [ucx_cv_have_aarch64_st64b_asm],
               [AC_COMPILE_IFELSE([AC_LANG_SOURCE([[
#ifndef __aarch64__
#error "ST64B is only supported on AArch64"
#endif

void test_st64b(void *dst, void *src)
{
    register void *src_reg asm("x8") = src;
    register void *dst_reg asm("x9") = dst;

    asm volatile(".arch_extension ls64\n"
                 "st64b x8, [x9]"
                 :
                 : "r"(src_reg), "r"(dst_reg)
                 : "memory");
}
]])],
               [ucx_cv_have_aarch64_st64b_asm=yes],
               [ucx_cv_have_aarch64_st64b_asm=no])])

AS_IF([test "x$ucx_cv_have_aarch64_st64b_asm" = "xyes"],
      [AC_DEFINE([HAVE_AARCH64_ST64B_ASM], [1],
                 [Define to 1 if AArch64 ST64B assembler is supported])])

AC_CONFIG_FILES([src/uct/ib/mlx5/Makefile src/uct/ib/mlx5/ucx-ib-mlx5.pc])

m4_include([src/uct/ib/mlx5/gdaki/configure.m4])

AC_DEFINE_UNQUOTED([uct_ib_mlx5_MODULES],
                   ["${uct_ib_mlx5_modules}"],
                   [IB MLX5 loadable modules])
