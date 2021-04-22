#
# Copyright (C) NVIDIA Corporation. 2021.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([UCX_CHECK_NVTX],[

AS_IF([test "x$nvtx_checked" != "xyes"],
   [
    AC_ARG_WITH([nvtx],
                [AS_HELP_STRING([--with-nvtx=(DIR)], [Enable the use of NVTX (default is no).])],
                [], [with_nvtx=no])

    AS_IF([test "x$with_nvtx" = "xno"],
        [
         nvtx_happy=no
        ],
        [
         save_CPPFLAGS="$CPPFLAGS"
         save_LDFLAGS="$LDFLAGS"
         save_LIBS="$LIBS"

         NVTX_CPPFLAGS=""
         NVTX_LDFLAGS=""
         NVTX_LIBS=""

         AS_IF([test ! -z "$with_nvtx" -a "x$with_nvtx" != "xyes" -a "x$with_nvtx" != "xguess"],
               [ucx_check_nvtx_dir="$with_nvtx"
                AS_IF([test -d "$with_nvtx/lib64"], [libsuff="64"], [libsuff=""])
                ucx_check_nvtx_libdir="$with_nvtx/lib$libsuff"
                NVTX_CPPFLAGS="-I$with_nvtx/include"
                NVTX_LDFLAGS="-L$ucx_check_nvtx_libdir -L$ucx_check_nvtx_libdir/stubs"])

         AS_IF([test ! -z "$with_nvtx_libdir" -a "x$with_nvtx_libdir" != "xyes"],
               [ucx_check_nvtx_libdir="$with_nvtx_libdir"
                NVTX_LDFLAGS="-L$ucx_check_nvtx_libdir -L$ucx_check_nvtx_libdir/stubs"])

         CPPFLAGS="$CPPFLAGS $NVTX_CPPFLAGS"
         LDFLAGS="$LDFLAGS $NVTX_LDFLAGS"

         # Check nvtx header files
         AC_CHECK_HEADERS([nvToolsExt.h],
                          [nvtx_happy="yes"], [nvtx_happy="no"])

         # Check nvtx libraries
         AS_IF([test "x$nvtx_happy" = "xyes"],
               [AC_CHECK_LIB([nvToolsExt], [nvtxRangePop],
                             [NVTX_LIBS="$NVTX_LIBS -lnvToolsExt"], [nvtx_happy="no"])])

         LDFLAGS="$save_LDFLAGS"
         CPPFLAGS="$save_CPPFLAGS"
         LDFLAGS="$save_LDFLAGS"
         LIBS="$save_LIBS"

         AS_IF([test "x$nvtx_happy" = "xyes"],
               [AC_SUBST([NVTX_CPPFLAGS], ["$NVTX_CPPFLAGS"])
                AC_SUBST([NVTX_LDFLAGS], ["$NVTX_LDFLAGS"])
                AC_SUBST([NVTX_LIBS], ["$NVTX_LIBS"])
                AC_DEFINE([HAVE_NVTX], 1, [Enable NVTX support])],
               [AS_IF([test "x$with_nvtx" != "xguess"],
                      [AC_MSG_ERROR([NVTX support is requested but nvtx packages cannot be found])],
                      [AC_MSG_WARN([NVTX not found])])])

        ]) # "x$with_nvtx" = "xno"

        nvtx_checked=yes
        AM_CONDITIONAL([HAVE_NVTX], [test "x$nvtx_happy" != xno])

   ]) # "x$nvtx_checked" != "xyes"

]) # UCX_CHECK_NVTX
