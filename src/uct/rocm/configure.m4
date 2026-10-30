#
# Copyright (C) Advanced Micro Devices, Inc. 2016 - 2018. ALL RIGHTS RESERVED.
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_ROCM

AS_IF([test "x$rocm_happy" = "xyes"], [uct_modules="${uct_modules}:rocm"])
uct_rocm_modules=""
AC_DEFINE_UNQUOTED([uct_rocm_MODULES], ["${uct_rocm_modules}"], [ROCM loadable modules])
AC_CONFIG_FILES([src/uct/rocm/Makefile
                 src/uct/rocm/ucx-rocm.pc])
