#
# Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

#
# Check library support for Infiniband Management Datagrams (MAD)
#
AC_ARG_WITH([mad],
            [AS_HELP_STRING([--with-mad=(DIR)],
                [Enable Infiniband MAD support (default is no).])],
            [],
            [with_mad=guess])

mad_happy=no
AS_IF([test "x$with_mad" = "xno"],
    [AC_MSG_WARN([Infiniband MAD support explicitly disabled])],

    [AS_CASE(["x$with_mad"],
        [x|xguess|xyes],
            [
                AC_MSG_NOTICE([Infiniband MAD Path not specified. Guessing ...])
                MAD_CFLAGS=""
                MAD_LDFLAGS=""
            ],
        [x/*],
            [
                AC_MSG_NOTICE([Infiniband MAD Path is "$with_mad" ...])
                MAD_CFLAGS="-I$with_mad/include"
                MAD_LDFLAGS="-L$with_mad/lib -L$with_mad/lib64"
            ],
        [AC_MSG_ERROR([Invalid Infiniband MAD path "$with_mad"])])

    save_CFLAGS="$CFLAGS"
    save_LDFLAGS="$LDFLAGS"

    CFLAGS="$CFLAGS $MAD_CFLAGS"
    LDFLAGS="$LDFLAGS $MAD_LDFLAGS"

    mad_happy=yes

    AC_CHECK_HEADER([infiniband/mad.h],        [:], [mad_happy=no])
    AC_CHECK_HEADER([infiniband/umad.h],       [:], [mad_happy=no])
    AC_CHECK_HEADER([infiniband/umad_types.h], [:], [mad_happy=no])

    AS_IF([test "x$mad_happy" = "xyes"],
        [
            AC_CHECK_LIB([ibmad],  [mad_build_pkt], [:], [mad_happy=no])
            AC_CHECK_LIB([ibumad], [umad_send],     [:], [mad_happy=no])
        ])

    CFLAGS="$save_CFLAGS"
    LDFLAGS="$save_LDFLAGS"

    AS_IF([test "x$mad_happy" = "xyes"],
        [
            AC_DEFINE([HAVE_MAD], 1, [Enable Infiniband MAD support])
            AC_SUBST([MAD_CFLAGS])
            AC_SUBST([MAD_LDFLAGS])
            AC_SUBST([MAD_LIBS], "-libmad -libumad")
        ],
        [
            AS_IF([test "x$with_mad" != "xguess"],
                [AC_MSG_ERROR([Infiniband MAD library or headers not found])],
                [AC_MSG_WARN([Disabling support for Infiniband MAD])])
        ])
    ]
)

AM_CONDITIONAL([HAVE_MAD], [test "x$mad_happy" = "xyes"])
