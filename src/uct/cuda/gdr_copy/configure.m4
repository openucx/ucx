#
# Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

#
# Check for GDRCOPY support
#
gdrcopy_happy="no"

AC_ARG_WITH([gdrcopy],
           [AS_HELP_STRING([--with-gdrcopy=(DIR)], [Enable the use of GDR_COPY (default is guess).])],
           [], [with_gdrcopy=guess])

AS_IF([test "x$with_gdrcopy" != "xno"],
    [save_CPPFLAGS="$CPPFLAGS"
     save_CFLAGS="$CFLAGS"
     save_LDFLAGS="$LDFLAGS"

     AS_IF([test ! -z "$with_gdrcopy" -a "x$with_gdrcopy" != "xyes" -a "x$with_gdrcopy" != "xguess"],
            [
            ucx_check_gdrcopy_dir="$with_gdrcopy"
            AS_IF([test -d "$with_gdrcopy/lib64"],[libsuff="64"],[libsuff=""])
            ucx_check_gdrcopy_libdir="$with_gdrcopy/lib$libsuff"
            CPPFLAGS="-I$with_gdrcopy/include $save_CPPFLAGS"
            LDFLAGS="-L$ucx_check_gdrcopy_libdir $save_LDFLAGS"
            ])
        AS_IF([test ! -z "$with_gdrcopy_libdir" -a "x$with_gdrcopy_libdir" != "xyes"],
            [ucx_check_gdrcopy_libdir="$with_nccl_libdir"
            LDFLAGS="-L$ucx_check_gdrcopy_libdir $save_LDFLAGS"])

        AC_CHECK_HEADERS([gdrapi.h],
            [AC_CHECK_LIB([gdrapi] , [gdr_pin_buffer],
                           [gdrcopy_happy="yes"],
                           [AC_MSG_WARN([GDR_COPY runtime not detected. Disable.])
                            gdrcopy_happy="no"])
            ], [gdrcopy_happy="no"])

        AS_IF([test "x$gdrcopy_happy" == "xyes"],
            [
                AC_DEFINE([HAVE_GDR_COPY], 1, [Enable GDR_COPY support])
                AC_SUBST(GDR_COPY_CPPFLAGS, "-I$ucx_check_gdrcopy_dir/include/ ")
                AC_SUBST(GDR_COPY_LDFLAGS, "-lgdrapi -L$ucx_check_gdrcopy_dir/lib64")
                CFLAGS="$save_CFLAGS $GDR_COPY_CFLAGS"
                CPPFLAGS="$save_CPPFLAGS $GDR_COPY_CPPFLAGS"
                LDFLAGS="$save_LDFLAGS $GDR_COPY_LDFLAGS"
            ],
            [
                AS_IF([test "x$with_gdrcopy" != "xguess"],
                    [AC_MSG_ERROR([gdrcopy support is requested but gdrcopy packages can't found])],
                    [AC_MSG_WARN([GDR_COPY not found])
                    AC_DEFINE([HAVE_GDR_COPY], [0], [Disable the use of GDR_COPY])])
            ])
    ],
    [AC_MSG_WARN([GDR_COPY was explicitly disabled])
    AC_DEFINE([HAVE_GDR_COPY], [0], [Disable the use of GDR_COPY])])

AM_CONDITIONAL([HAVE_GDR_COPY], [test "x$gdrcopy_happy" != xno])
