#
# Copyright (C) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_MUSA

AS_IF([test "x$musa_happy" = "xyes"],
      [uct_modules="${uct_modules}:musa"])

uct_musa_modules=""
AC_DEFINE_UNQUOTED([uct_musa_MODULES], ["${uct_musa_modules}"],
                   [MUSA loadable modules])
AC_CONFIG_FILES([src/uct/musa/Makefile
                 src/uct/musa/ucx-musa.pc])
