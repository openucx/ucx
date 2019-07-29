#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

UCX_CHECK_ROCM
AS_IF([test "x$rocm_happy" = "xyes"], [ucm_modules="${ucm_modules}:rocm"])
AC_CONFIG_FILES([src/ucm/rocm/Makefile])
