#
# Copyright (c) Intel Corporation, 2023. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([UCX_CHECK_ZE],[

AS_IF([test "x$ze_checked" != "xyes"],
   [
    AC_ARG_WITH([ze],
                [AS_HELP_STRING([--with-ze=(DIR)], [Enable the use of ZE (default is guess).])],
                [], [with_ze=guess])

    AS_IF([test "x$with_ze" = "xno"],
        [
         ze_happy="no"
        ],
        [
         save_CPPFLAGS="$CPPFLAGS"
         save_LDFLAGS="$LDFLAGS"
         save_LIBS="$LIBS"

         ZE_CPPFLAGS=""
         ZE_LDFLAGS=""
         ZE_LIBS=""

         AS_IF([test ! -z "$with_ze" -a "x$with_ze" != "xyes" -a "x$with_ze" != "xguess"],
               [ucx_check_ze_dir="$with_ze"
                AS_IF([test -d "$with_ze/lib64"], [libsuff="64"], [libsuff=""])
                ucx_check_ze_libdir="$with_ze/lib$libsuff"
                ZE_CPPFLAGS="-I$with_ze/include"
                ZE_LDFLAGS="-L$ucx_check_ze_libdir -L$ucx_check_ze_libdir/stubs"])

         AS_IF([test ! -z "$with_ze_libdir" -a "x$with_ze_libdir" != "xyes"],
               [ucx_check_ze_libdir="$with_ze_libdir"
                ZE_LDFLAGS="-L$ucx_check_ze_libdir -L$ucx_check_ze_libdir/stubs"])

         CPPFLAGS="$CPPFLAGS $ZE_CPPFLAGS"
         LDFLAGS="$LDFLAGS $ZE_LDFLAGS"

         # Check ze header files
         AC_CHECK_HEADERS([level_zero/ze_api.h],
                          [ze_happy="yes"], [ze_happy="no"])

         # Check ze libraries
         AS_IF([test "x$ze_happy" = "xyes"],
               [AC_CHECK_LIB([ze_loader], [zeInit],
                             [ZE_LIBS="$ZE_LIBS -lze_loader"], [ze_happy="no"])])

         CPPFLAGS="$save_CPPFLAGS"
         LDFLAGS="$save_LDFLAGS"
         LIBS="$save_LIBS"

         AS_IF([test "x$ze_happy" = "xyes"],
               [AC_SUBST([ZE_CPPFLAGS], ["$ZE_CPPFLAGS"])
                AC_SUBST([ZE_LDFLAGS], ["$ZE_LDFLAGS"])
                AC_SUBST([ZE_LIBS], ["$ZE_LIBS"])
                AC_DEFINE([HAVE_ZE], 1, [Enable ZE support])],
               [AS_IF([test "x$with_ze" != "xguess"],
                      [AC_MSG_ERROR([ZE support is requested but ze packages cannot be found])],
                      [AC_MSG_WARN([ZE not found])])])

        ]) # "x$with_ze" = "xno"

        ze_checked=yes
        AM_CONDITIONAL([HAVE_ZE], [test "x$ze_happy" != xno])

   ]) # "x$ze_checked" != "xyes"

]) # UCX_CHECK_ZE
