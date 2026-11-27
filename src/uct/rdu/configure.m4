#
# Copyright (C) SambaNova Systems, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_RDU

AS_IF([test "x$rdu_happy" = "xyes"], [uct_modules="${uct_modules}:rdu"])
AC_CONFIG_FILES([src/uct/rdu/Makefile
                 src/uct/rdu/ucx-rdu.pc])
