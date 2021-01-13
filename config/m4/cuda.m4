#
# Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([UCX_CHECK_CUDA],[

AS_IF([test "x$cuda_checked" != "xyes"],
   [
    AC_ARG_WITH([cuda],
                [AS_HELP_STRING([--with-cuda=(DIR)], [Enable the use of CUDA (default is guess).])],
                [], [with_cuda=guess])

    AS_IF([test "x$with_cuda" = "xno"],
        [
         cuda_happy=no
         have_cuda_static=no
        ],
        [
         save_CPPFLAGS="$CPPFLAGS"
         save_LDFLAGS="$LDFLAGS"
         save_LIBS="$LIBS"

         CUDA_CPPFLAGS=""
         CUDA_LDFLAGS=""
         CUDA_LIBS=""
         CUDA_STATIC_LIBS=""

         AS_IF([test ! -z "$with_cuda" -a "x$with_cuda" != "xyes" -a "x$with_cuda" != "xguess"],
               [ucx_check_cuda_dir="$with_cuda"
                AS_IF([test -d "$with_cuda/lib64"], [libsuff="64"], [libsuff=""])
                ucx_check_cuda_libdir="$with_cuda/lib$libsuff"
                CUDA_CPPFLAGS="-I$with_cuda/include"
                CUDA_LDFLAGS="-L$ucx_check_cuda_libdir -L$ucx_check_cuda_libdir/stubs"])

         AS_IF([test ! -z "$with_cuda_libdir" -a "x$with_cuda_libdir" != "xyes"],
               [ucx_check_cuda_libdir="$with_cuda_libdir"
                CUDA_LDFLAGS="-L$ucx_check_cuda_libdir -L$ucx_check_cuda_libdir/stubs"])

         CPPFLAGS="$CPPFLAGS $CUDA_CPPFLAGS"
         LDFLAGS="$LDFLAGS $CUDA_LDFLAGS"

         # Check cuda header files
         AC_CHECK_HEADERS([cuda.h cuda_runtime.h],
                          [cuda_happy="yes"], [cuda_happy="no"])

         # Check cuda libraries
         AS_IF([test "x$cuda_happy" = "xyes"],
               [AC_CHECK_LIB([cuda], [cuDeviceGetUuid],
                             [CUDA_LIBS="$CUDA_LIBS -lcuda"], [cuda_happy="no"])])
         AS_IF([test "x$cuda_happy" = "xyes"],
               [AC_CHECK_LIB([cudart], [cudaGetDeviceCount],
                             [CUDA_LIBS="$CUDA_LIBS -lcudart"], [cuda_happy="no"])])

         LDFLAGS="$save_LDFLAGS"

         # Check for cuda static library
         have_cuda_static="no"
         AS_IF([test "x$cuda_happy" = "xyes"],
               [AC_CHECK_LIB([cudart_static], [cudaGetDeviceCount],
                             [CUDA_STATIC_LIBS="$CUDA_STATIC_LIBS -lcudart_static"
                              have_cuda_static="yes"],
                             [], [-ldl -lrt -lpthread])])

         CPPFLAGS="$save_CPPFLAGS"
         LDFLAGS="$save_LDFLAGS"
         LIBS="$save_LIBS"

         AS_IF([test "x$cuda_happy" = "xyes"],
               [AC_SUBST([CUDA_CPPFLAGS], ["$CUDA_CPPFLAGS"])
                AC_SUBST([CUDA_LDFLAGS], ["$CUDA_LDFLAGS"])
                AC_SUBST([CUDA_LIBS], ["$CUDA_LIBS"])
                AC_SUBST([CUDA_STATIC_LIBS], ["$CUDA_STATIC_LIBS"])
                AC_DEFINE([HAVE_CUDA], 1, [Enable CUDA support])],
               [AS_IF([test "x$with_cuda" != "xguess"],
                      [AC_MSG_ERROR([CUDA support is requested but cuda packages cannot be found])],
                      [AC_MSG_WARN([CUDA not found])])])

        ]) # "x$with_cuda" = "xno"

        cuda_checked=yes
        AM_CONDITIONAL([HAVE_CUDA], [test "x$cuda_happy" != xno])
        AM_CONDITIONAL([HAVE_CUDA_STATIC], [test "X$have_cuda_static" = "Xyes"])

   ]) # "x$cuda_checked" != "xyes"

]) # UCX_CHECK_CUDA
