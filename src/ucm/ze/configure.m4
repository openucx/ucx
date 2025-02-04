#
# Copyright (C) Intel Corporation, 2023. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

UCX_CHECK_ZE
AS_IF([test "x$ze_happy" = "xyes"], [ucm_modules="${ucm_modules}:ze"])
AC_CONFIG_FILES([src/ucm/ze/Makefile])
