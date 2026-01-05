#
# Copyright (c) Intel Corporation, 2025. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([UCX_CHECK_GAUDI],[

AS_IF([test "x$gaudi_checked" != "xyes"],
   [
    AC_ARG_WITH([gaudi],
                [AS_HELP_STRING([--with-gaudi=(DIR)], [Enable the use of GAUDI (default is guess).])],
                [], [with_gaudi=guess])

    AS_IF([test "x$with_gaudi" = "xno"],
        [
         gaudi_happy="no"
        ],
        [
         save_CPPFLAGS="$CPPFLAGS"
         save_LDFLAGS="$LDFLAGS"
         save_LIBS="$LIBS"

         GAUDI_CPPFLAGS=""
         GAUDI_LDFLAGS=""
         GAUDI_LIBS=""

         AS_IF([test ! -z "$with_gaudi" -a "x$with_gaudi" != "xyes" -a "x$with_gaudi" != "xguess"],
               [ucx_check_gaudi_dir="$with_gaudi"
                ucx_check_gaudi_libdir="$with_gaudi/lib/habanalabs"
                GAUDI_CPPFLAGS="-I$with_gaudi/include/habanalabs -I/usr/include/drm -I/usr/include/libdrm"
                GAUDI_LDFLAGS="-L$ucx_check_gaudi_libdir"],
               [GAUDI_CPPFLAGS="-I/usr/include/habanalabs -I/usr/include/drm -I/usr/include/libdrm"
                GAUDI_LDFLAGS="-L/usr/lib/habanalabs"])

         AS_IF([test ! -z "$with_gaudi_libdir" -a "x$with_gaudi_libdir" != "xyes"],
               [ucx_check_gaudi_libdir="$with_gaudi_libdir"
                GAUDI_LDFLAGS="-L$ucx_check_gaudi_libdir"])

         CPPFLAGS="$CPPFLAGS $GAUDI_CPPFLAGS"
         LDFLAGS="$LDFLAGS $GAUDI_LDFLAGS"

         # Check gaudi header files
         AC_CHECK_HEADERS([hlthunk.h],
                          [gaudi_happy="yes"], [gaudi_happy="no"])

         # Check gaudi libraries
         AS_IF([test "x$gaudi_happy" = "xyes"],
               [AC_CHECK_LIB([hl-thunk], [hlthunk_open],
                             [GAUDI_LIBS="$GAUDI_LIBS -lhl-thunk -lscal -lSynapse -lSynapseMme"], [gaudi_happy="no"])])

         CPPFLAGS="$save_CPPFLAGS"
         LDFLAGS="$save_LDFLAGS"
         LIBS="$save_LIBS"

         AS_IF([test "x$gaudi_happy" = "xyes"],
               [AC_SUBST([GAUDI_CPPFLAGS], ["$GAUDI_CPPFLAGS"])
                AC_SUBST([GAUDI_LDFLAGS], ["$GAUDI_LDFLAGS"])
                AC_SUBST([GAUDI_LIBS], ["$GAUDI_LIBS"])
                AC_DEFINE([HAVE_GAUDI], 1, [Enable GAUDI support])],
               [AS_IF([test "x$with_gaudi" != "xguess"],
                      [AC_MSG_ERROR([GAUDI support is requested but gaudi packages cannot be found])],
                      [AC_MSG_WARN([GAUDI not found])])])

        ]) # "x$with_gaudi" = "xno"

        gaudi_checked=yes
        AM_CONDITIONAL([HAVE_GAUDI], [test "x$gaudi_happy" != xno])

   ]) # "x$gaudi_checked" != "xyes"

]) # UCX_CHECK_GAUDI
