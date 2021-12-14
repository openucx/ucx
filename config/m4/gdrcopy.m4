#
# Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
# Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([UCX_CHECK_GDRCOPY],[

AS_IF([test "x$gdrcopy_checked" != "xyes"],[

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
            [ucx_check_gdrcopy_libdir="$with_gdrcopy_libdir"
            LDFLAGS="-L$ucx_check_gdrcopy_libdir $save_LDFLAGS"])

        AC_CHECK_HEADERS([gdrapi.h],
            [AC_CHECK_LIB([gdrapi] , [gdr_pin_buffer],
                           [gdrcopy_happy="yes"],
                           [AC_MSG_WARN([GDR_COPY runtime not detected. Disable.])
                            gdrcopy_happy="no"])
            ], [gdrcopy_happy="no"])

        AS_IF([test "x$gdrcopy_happy" = "xyes"],
            [AC_CHECK_DECLS([gdr_copy_to_mapping], [], [], [#include "gdrapi.h"])])

        CFLAGS="$save_CFLAGS"
        CPPFLAGS="$save_CPPFLAGS"
        LDFLAGS="$save_LDFLAGS"

        AS_IF([test "x$gdrcopy_happy" = "xyes"],
            [
                AC_SUBST(GDR_COPY_CPPFLAGS, "-I$ucx_check_gdrcopy_dir/include/ ")

                gdr_copy_ldflags="-lgdrapi"
                AS_IF([test ! -z "$ucx_check_gdrcopy_libdir"],
                    gdr_copy_ldflags="$gdr_copy_ldflags -L$ucx_check_gdrcopy_libdir")
                AC_SUBST(GDR_COPY_LDFLAGS, "$gdr_copy_ldflags")
            ],
            [
                AS_IF([test "x$with_gdrcopy" != "xguess"],
                    [AC_MSG_ERROR([gdrcopy support is requested but gdrcopy packages cannot be found])],
                    [AC_MSG_WARN([GDR_COPY not found])])
            ])

    ],
    [AC_MSG_WARN([GDR_COPY was explicitly disabled])])

gdrcopy_checked=yes
AM_CONDITIONAL([HAVE_GDR_COPY], [test "x$gdrcopy_happy" != xno])

])

])
