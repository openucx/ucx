#
# Copyright (C) SambaNova Systems, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_RDU

# RDU device memory is registered by the IB memory domain, so the RDU memory
# domain is useful only together with IB.
AS_IF([test "x$rdu_happy" = "xyes" -a "x$with_ib" = "xyes"],
      [uct_modules="${uct_modules}:rdu"
       have_rdu=yes
       AC_DEFINE([HAVE_RDU], 1, [Enable SambaNova RDU support])],
      [have_rdu=no
       AS_IF([test "x$with_rdu" != "xguess" -a "x$with_rdu" != "xno"],
             [AC_MSG_ERROR([RDU support was requested but IB support is not available])])])

AM_CONDITIONAL([HAVE_RDU], [test "x$have_rdu" = "xyes"])

AC_CONFIG_FILES([src/uct/rdu/Makefile
                 src/uct/rdu/ucx-rdu.pc])
