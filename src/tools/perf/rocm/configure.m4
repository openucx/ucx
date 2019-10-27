#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

UCX_CHECK_ROCM

AS_IF([test "x$rocm_happy" = "xyes"], [ucx_perftest_modules="${ucx_perftest_modules}:rocm"])

AC_CONFIG_FILES([src/tools/perf/rocm/Makefile])
