#
# Copyright (C) NVIDIA Corporation. 2021.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

nvtx_happy="no"
AC_ARG_WITH([nvtx],
            [AC_HELP_STRING([--with-nvtx=(PATH)],
                            [Enable the use of NVTX (default is guess).])
            ], [], [with_nvtx=guess])

AS_IF([test "x$with_nvtx" != "xno"],
    [save_CPPFLAGS="$CPPFLAGS"

     AS_IF([test ! -z "$with_nvtx" -a "x$with_nvtx" != "xyes" -a "x$with_nvtx" != "xguess"],
            [
            ucx_check_nvtx_dir="$with_nvtx"
            CPPFLAGS="-I$with_nvtx/include/nvtx3 $save_CPPFLAGS"
            ])

        AC_CHECK_HEADERS([nvToolsExt.h],
            [nvtx_happy="yes"], [nvtx_happy="no"])

        CPPFLAGS="$save_CPPFLAGS"

        AS_IF([test "x$nvtx_happy" = "xyes"],
            [
                AC_SUBST(NVTX_CPPFLAGS, "-I$ucx_check_nvtx_dir/include/nvtx3 ")
            ],
            [
                AS_IF([test "x$with_nvtx" != "xguess"],
                    [AC_MSG_ERROR([nvtx support is requested but nvtx headers cannot be found])],
                    [AC_MSG_WARN([NVTX not found])])
            ])

    ],
    [AC_MSG_WARN([NVTX was explicitly disabled])])

AS_IF([test "x$nvtx_happy" = "xyes"],
	[AS_MESSAGE([enabling nvtx profiling])
	 AC_DEFINE([HAVE_NVTX], [1], [Enable NVTX profiling])
	 HAVE_NVTX=yes]
	[:]
)

AM_CONDITIONAL([HAVE_NVTX],[test "x$HAVE_NVTX" = "xyes"])
