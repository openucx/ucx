#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

knem_happy="no"

AC_ARG_WITH([knem],
            [AS_HELP_STRING([--with-knem=(DIR)], [Enable the use of KNEM (default is guess).])],
            [], [with_knem=guess])

AS_IF([test "x$with_knem" != xno],
      [AS_IF([test "x$with_knem" = "xguess" -o "x$with_knem" = xyes -o "x$with_knem" = "x"],
             [AC_MSG_NOTICE([KNEM path was not found, guessing ...])
              ucx_check_knem_include_dir=$(pkg-config --cflags knem)],
             [ucx_check_knem_include_dir=-I$with_knem/include])

     save_CPPFLAGS="$CPPFLAGS"
     CPPFLAGS="$ucx_check_knem_include_dir $CPPFLAGS"

     AC_CHECK_DECL([KNEM_CMD_GET_INFO],
                   [AC_SUBST([KNEM_CPPFLAGS], [$ucx_check_knem_include_dir])
                    uct_modules="${uct_modules}:knem"
                    knem_happy="yes"],
                   [AS_IF([test "x$with_knem" != xguess],
                          [AC_MSG_ERROR([KNEM requested but required file (knem_io.h) could not be found])],
                          [AC_MSG_WARN([KNEM requested but required file (knem_io.h) could not be found])])],
                   [[#include <knem_io.h>]])

     CPPFLAGS="$save_CPPFLAGS"

    ],
    [AC_MSG_WARN([KNEM was explicitly disabled])]
)

AM_CONDITIONAL([HAVE_KNEM], [test "x$knem_happy" != xno])
AC_CONFIG_FILES([src/uct/sm/scopy/knem/Makefile
                 src/uct/sm/scopy/knem/ucx-knem.pc])
