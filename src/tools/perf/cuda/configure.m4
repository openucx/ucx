#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

UCX_CHECK_CUDA

AS_IF([test "x$cuda_happy" = "xyes"], [ucx_perftest_modules="${ucx_perftest_modules}:cuda"])

AC_CONFIG_FILES([src/tools/perf/cuda/Makefile])
