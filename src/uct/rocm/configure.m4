#
# Copyright (C) Advanced Micro Devices, Inc. 2016 - 2018. ALL RIGHTS RESERVED.
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_ROCM

AS_IF([test "x$rocm_happy" == "xyes"], [uct_modules+=":rocm"])
AC_CONFIG_FILES([src/uct/rocm/Makefile])
