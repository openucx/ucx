#
# Copyright (C) SambaNova Systems, 2026. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

UCX_CHECK_RDU

AS_IF([test "x$rdu_happy" = "xyes"], [ucx_perftest_modules="${ucx_perftest_modules}:rdu"])

AC_CONFIG_FILES([src/tools/perf/rdu/Makefile])
