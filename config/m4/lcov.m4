#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

#
# Enable LCOV capable build
#

AC_ARG_ENABLE([lcov],
    AS_HELP_STRING([--enable-lcov], [Enable code coverage reporting]),
    [],
    [enable_lcov=no])

lcov_happy=no
AS_IF([test "x$enable_lcov" = xno],
    [AC_MSG_WARN([LCOV support is not enabled.])],
    [
        AC_CHECK_PROG([GCOVBIN], gcov, gcov, notfound)
        AC_CHECK_PROG([LCOVBIN], lcov, lcov, notfound)
        AC_CHECK_PROG([GENHTMLBIN], genhtml, genhtml, notfound)

        lcov_happy=yes
        AS_IF([test "x$GCOVBIN" = xnotfound], lcov_happy=no)
        AS_IF([test "x$LCOVBIN" = xnotfound], lcov_happy=no)
        AS_IF([test "x$GENHTMLBIN" = xnotfound], lcov_happy=no)

        AS_IF([test "x$lcov_happy" != xyes],
            [AC_MSG_ERROR([LCOV requested: gcov, lcov or genhtml not found])])
    ])

AM_CONDITIONAL([HAVE_LCOV], [test "x$lcov_happy" = xyes])
