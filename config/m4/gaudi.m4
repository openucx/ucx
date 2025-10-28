#
# Copyright (c) Intel Corporation, 2023. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#
AC_DEFUN([UCX_CHECK_GAUDI],[

AS_IF([test "x$gaudi_checked" != "xyes"],
   [
    AC_ARG_WITH([gaudi],
                [AS_HELP_STRING([--with-gaudi=(DIR)], [Enable the use of Gaudi (default is guess).])],
                [], [with_gaudi=guess])
    AS_IF([test "x$with_gaudi" = "xno"],
        [
         gaudi_happy="no"
        ],
        [
        AS_CASE(["x$with_gaudi"],
        [x|xguess|xyes],
        [AC_MSG_NOTICE([Gaudi path was not specified. Guessing ...])
            GAUDI_CPPFLAGS="-I/usr/include/habanalabs -I/usr/include/drm -I/usr/include/libdrm"
            GAUDI_LDFLAGS="-L/usr/lib/habanalabs"
            GAUDI_LIBS="-lhl-thunk -lscal -lSynapse -lSynapseMme"],
        [x/*],
        [AC_MSG_NOTICE([Gaudi path given as $with_gaudi ...])
            GAUDI_CPPFLAGS="-I$with_gaudi/include/habanalabs -I/usr/include/drm -I/usr/include/libdrm"
            GAUDI_LDFLAGS="-L$with_gaudi/lib/habanalabs"
            GAUDI_LIBS="-lhl-thunk -lscal -lSynapse -lSynapseMme"],
         )

         save_CPPFLAGS="$CPPFLAGS"
         save_LDFLAGS="$LDFLAGS"
         save_LIBS="$LIBS"

         CPPFLAGS="$GAUDI_CPPFLAGS $CPPFLAGS"
         LDFLAGS="$GAUDI_LDFLAGS $LDFLAGS"
         LIBS="$GAUDI_LIBS $LIBS"

         gaudi_happy=yes
         AC_CHECK_HEADERS([hlthunk.h], [gaudi_happy=yes], [gaudi_happy=no])
         AS_IF([test "x$gaudi_happy" = "xyes"],
               [AC_CHECK_LIB([hl-thunk], [hlthunk_open],[gaudi_happy=yes], [gaudi_happy="no"])])

         CPPFLAGS="$save_CPPFLAGS"
         LDFLAGS="$save_LDFLAGS"
         LIBS="$save_LIBS"

         AS_IF([test "x$gaudi_happy" = "xyes"],
               [AC_SUBST([GAUDI_CPPFLAGS], ["$GAUDI_CPPFLAGS"])
                AC_SUBST([GAUDI_LDFLAGS], ["$GAUDI_LDFLAGS"])
                AC_SUBST([GAUDI_LIBS], ["$GAUDI_LIBS"])
                AC_DEFINE([HAVE_GAUDI], 1, [Enable GAUDI support])],
               [AS_IF([test "x$with_gaudi" != "xguess"],
                      [AC_MSG_ERROR([Gaudi support is requested but Gaudi packages cannot be found])],
                      [AC_MSG_WARN([Gaudi not found])])])

        ]) # "x$with_gaudi" = "xno"

    gaudi_checked=yes
    AM_CONDITIONAL([HAVE_GAUDI], [test "x$gaudi_happy" != xno])

   ]) # "x$gaudi_checked" != "xyes"

]) # UCX_CHECK_GAUDI
