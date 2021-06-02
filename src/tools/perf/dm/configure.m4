#
# Copyright (C) Huawei Technologies Co., Ltd. 2021.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

AS_IF([test "x$has_ib_dm" = "xyes"], [ucx_perftest_modules="${ucx_perftest_modules}:dm"])

AC_CONFIG_FILES([src/tools/perf/dm/Makefile])
