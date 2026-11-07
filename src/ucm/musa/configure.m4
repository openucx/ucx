#
# Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_MUSA
AS_IF([test "x$musa_happy" = "xyes"],
      [ucm_modules="${ucm_modules}:musa"])
AC_CONFIG_FILES([src/ucm/musa/Makefile])
