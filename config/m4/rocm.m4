#
# Copyright (C) Advanced Micro Devices, Inc. 2016 - 2018. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

# ROCM_PARSE_FLAGS(ARG, VAR_LIBS, VAR_LDFLAGS, VAR_CPPFLAGS)
# ----------------------------------------------------------
# Parse whitespace-separated ARG into appropriate LIBS, LDFLAGS, and
# CPPFLAGS variables.
AC_DEFUN([ROCM_PARSE_FLAGS],
[for arg in $$1 ; do
    AS_CASE([$arg],
        [yes],               [],
        [no],                [],
        [-l*|*.a|*.so],      [$2="$$2 $arg"],
        [-L*|-WL*|-Wl*],     [$3="$$3 $arg"],
        [-I*],               [$4="$$4 $arg"],
        [*lib|*lib/|*lib64|*lib64/],[AS_IF([test -d $arg], [$3="$$3 -L$arg"],
                                 [AC_MSG_WARN([$arg of $1 not parsed])])],
        [*include|*include/],[AS_IF([test -d $arg], [$4="$$4 -I$arg"],
                                 [AC_MSG_WARN([$arg of $1 not parsed])])],
        [AC_MSG_WARN([$arg of $1 not parsed])])
done])

#
# Check for ROCm  support
#
AC_ARG_WITH([rocm],
    [AS_HELP_STRING([--with-rocm=(DIR)],
        [Enable the use of ROCm (default is autodetect).])],
    [],
    [with_rocm=guess])

rocm_happy=no
AS_IF([test "x$with_rocm" != "xno"],
    [AS_CASE(["x$with_rocm"],
        [x|xguess|xyes],
            [AC_MSG_NOTICE([ROCm path was not specified. Guessing ...])
             with_rocm=/opt/rocm
             ROCM_CPPFLAGS="-I$with_rocm/libhsakmt/include/libhsakmt -I$with_rocm/include/hsa -I$with_rocm/include"
             ROCM_LDFLAGS="-L$with_rocm/hsa/lib -L$with_rocm/lib"
             ROCM_LIBS="-lhsa-runtime64"],
        [x/*],
            [AC_MSG_NOTICE([ROCm path given as $with_rocm ...])
             ROCM_CPPFLAGS="-I$with_rocm/libhsakmt/include/libhsakmt -I$with_rocm/include/hsa -I$with_rocm/include"
             ROCM_LDFLAGS="-L$with_rocm/hsa/lib -L$with_rocm/lib"
             ROCM_LIBS="-lhsa-runtime64"],
        [AC_MSG_NOTICE([ROCm flags given ...])
         ROCM_PARSE_FLAGS([with_rocm],
                          [ROCM_LIBS], [ROCM_LDFLAGS], [ROCM_CPPFLAGS])])
    SAVE_CPPFLAGS="$CPPFLAGS"
    SAVE_LDFLAGS="$LDFLAGS"
    SAVE_LIBS="$LIBS"
    CPPFLAGS="$ROCM_CPPFLAGS $CPPFLAGS"
    LDFLAGS="$ROCM_LDFLAGS $LDFLAGS"
    LIBS="$ROCM_LIBS $LIBS"
    rocm_happy=yes
    AS_IF([test "x$rocm_happy" = xyes],
          [AC_CHECK_HEADERS([hsa.h], [rocm_happy=yes], [rocm_happy=no])])
    AS_IF([test "x$rocm_happy" = xyes],
          [AC_CHECK_HEADERS([hsa_ext_amd.h], [rocm_happy=yes], [rocm_happy=no])])
    AS_IF([test "x$rocm_happy" = xyes],
          [AC_CHECK_HEADERS([hsakmt.h], [rocm_happy=yes], [rocm_happy=no])])
    AS_IF([test "x$rocm_happy" = xyes],
          [AC_CHECK_DECLS([hsaKmtProcessVMRead,hsaKmtProcessVMWrite],
              [rocm_happy=yes],
              [rocm_happy=no
               AC_MSG_WARN([ROCm without CMA support was detected. Disable.])],
              [#include <hsakmt.h>])])
    AS_IF([test "x$rocm_happy" = xyes],
          [AC_SEARCH_LIBS([hsa_init], [hsa-runtime64])
           AS_CASE(["x$ac_cv_search_hsa_init"],
               [xnone*], [],
               [xno], [rocm_happy=no],
               [x-l*], [ROCM_LIBS="$ac_cv_search_hsa_init $ROCM_LIBS"])])
    AS_IF([test "x$rocm_happy" == "xyes"],
          [AC_DEFINE([HAVE_ROCM], [1], [Set to 1 to enable ROCm support])
           transports="${transports},rocm"
           AC_SUBST([ROCM_CPPFLAGS])
           AC_SUBST([ROCM_LDFLAGS])
           AC_SUBST([ROCM_LIBS])],
          [AC_DEFINE([HAVE_ROCM], [0], [Set to 1 to enable ROCm support])
           AC_MSG_WARN([ROCm not found])])
    CPPFLAGS="$SAVE_CPPFLAGS"
    LDFLAGS="$SAVE_LDFLAGS"
    LIBS="$SAVE_LIBS"
    ],
    [AC_DEFINE([HAVE_ROCM], [0], [Set to 1 to enable ROCm support])
     AC_MSG_WARN([ROCm was explicitly disabled])]
)

AM_CONDITIONAL([HAVE_ROCM], [test "x$rocm_happy" != xno])

