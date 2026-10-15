#
# Copyright (C) SambaNova Systems, 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_RDU
AS_IF([test "x$rdu_happy" = "xyes"], [ucm_modules="${ucm_modules}:rdu"])
AC_CONFIG_FILES([src/ucm/rdu/Makefile])
AM_CONDITIONAL([HAVE_UCM_RDU], [test "x$rdu_happy" = "xyes"])
