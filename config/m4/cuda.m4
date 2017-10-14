#
# Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

#
# Check for CUDA support
#
cuda_happy="no"

AC_ARG_WITH([cuda],
           [AS_HELP_STRING([--with-cuda=(DIR)], [Enable the use of CUDA (default is guess).])],
           [], [with_cuda=guess])

AS_IF([test "x$with_cuda" != "xno"],
        [save_CPPFLAGS="$CPPFLAGS"
        save_CFLAGS="$CFLAGS"
        save_LDFLAGS="$LDFLAGS"

        AS_IF([test ! -z "$with_cuda" -a "x$with_cuda" != "xyes" -a "x$with_cuda" != "xguess"],
            [
                ucx_check_cuda_dir="$with_cuda"
                AS_IF([test -d "$with_cuda/lib64"],[libsuff="64"],[libsuff=""])
                ucx_check_cuda_libdir="$with_cuda/lib$libsuff"
                CPPFLAGS="-I$with_cuda/include $save_CPPFLAGS"
                LDFLAGS="-L$ucx_check_cuda_libdir $save_LDFLAGS"
            ])

        AS_IF([test ! -z "$with_cuda_libdir" -a "x$with_cuda_libdir" != "xyes"],
            [ucx_check_cuda_libdir="$with_cuda_libdir"
            LDFLAGS="-L$ucx_check_cuda_libdir $save_LDFLAGS"])

        AC_CHECK_HEADERS([cuda.h cuda_runtime.h],
            [AC_CHECK_LIB([cuda] , [cuPointerGetAttribute],
                [cuda_happy="yes"],
                [AC_MSG_WARN([CUDA runtime not detected. Disable.])
                cuda_happy="no"])
            ],[cuda_happy="no"])


        AS_IF([test "x$cuda_happy" == "xyes"],
                [
                    AC_DEFINE([HAVE_CUDA], 1, [Enable CUDA support])
                    AC_SUBST(CUDA_CPPFLAGS, "-I$ucx_check_cuda_dir/include ")
                    AC_SUBST(CUDA_LDFLAGS, "-lcudart -lcuda -L$ucx_check_cuda_libdir/ ")
                    CFLAGS="$save_CFLAGS $CUDA_CFLAGS"
                    CPPFLAGS="$save_CPPFLAGS $CUDA_CPPFLAGS"
                    LDFLAGS="$saveLDFLAGS $CUDA_LDFLAGS"
                ],
                [
                    AS_IF([test "x$with_cuda" != "xguess"],
                        [AC_MSG_ERROR([CUDA support is requested but cuda packages can't found])],
                        [AC_MSG_WARN([CUDA not found])])
                    AC_DEFINE([HAVE_CUDA], [0], [Disable the use of CUDA])
                ])
    ],
    [AC_MSG_WARN([CUDA was explicitly disabled])
    AC_DEFINE([HAVE_CUDA], [0], [Disable the use of CUDA])]
)


AM_CONDITIONAL([HAVE_CUDA], [test "x$cuda_happy" != xno])
