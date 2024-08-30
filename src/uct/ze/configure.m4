#
# Copyright (C) Intel Corporation, 2023-2024. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_ZE

AS_IF([test "x$ze_happy" = "xyes"], [uct_modules="${uct_modules}:ze"])
AC_CONFIG_FILES([src/uct/ze/Makefile
                 src/uct/ze/ucx-ze.pc])
