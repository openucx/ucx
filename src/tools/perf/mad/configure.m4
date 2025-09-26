#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

AS_IF([test "x$mad_happy" = "xyes"], [ucx_perftest_modules="${ucx_perftest_modules}:mad"])

AC_CONFIG_FILES([src/tools/perf/mad/Makefile])
