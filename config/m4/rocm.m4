#
# Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

#
# Check for ROCm  support
#
rocm_happy="no"

AC_ARG_WITH([rocm],
           [AS_HELP_STRING([--with-rocm=(DIR)], [Enable the use of ROCm (default is autodetect).])],
           [], [with_rocm=guess])

AS_IF([test "x$with_rocm" != "xno"],

      [AS_IF([test "x$with_rocm" == "x" || test "x$with_rocm" == "xguess" || test "x$with_rocm" == "xyes"],
             [
              AC_MSG_NOTICE([ROCm path was not specified. Guessing ...])
              with_rocm=/opt/rocm
              ],
              [:])
        AC_CHECK_HEADERS([$with_rocm/include/hsa/hsa_ext_amd.h],
                       [AC_CHECK_DECLS([hsaKmtProcessVMRead,hsaKmtProcessVMWrite],
                           [rocm_happy="yes"],
                           [AC_MSG_WARN([ROCm without CMA support was detected. Disable.])
                            rocm_happy="no"],
                            [#include <$with_rocm/include/libhsakmt/hsakmt.h>])
                           AS_IF([test "x$rocm_happy" == "xyes"],
                            [AC_DEFINE([HAVE_ROCM], 1, [Enable ROCm support])
                             transports="${transports},rocm"
                            AC_SUBST(ROCM_CPPFLAGS, "-I$with_rocm/include/hsa -I$with_rocm/include/libhsakmt -DHAVE_ROCM=1")
                            AC_SUBST(ROCM_CFLAGS, "-I$with_rocm/include/hsa -I$with_rocm/include/libhsakmt -DHAVE_ROCM=1")
                            AC_SUBST(ROCM_LDFLAGS, "-lhsa-runtime64 -L$with_rocm/lib")
                            CFLAGS="$CFLAGS $ROCM_CFLAGS"
                            CPPFLAGS="$CPPFLAGS $ROCM_CPPFLAGS"
                            LDFLAGS="$LDFLAGS $ROCM_LDFLAGS"],
                        [])],
                       [AC_MSG_WARN([ROCm not found])
                        AC_DEFINE([HAVE_ROCM], [0], [Disable the use of ROCm])])],
      [AC_MSG_WARN([ROCm was explicitly disabled])
      AC_DEFINE([HAVE_ROCM], [0], [Disable the use of ROCm])]
)


AM_CONDITIONAL([HAVE_ROCM], [test "x$rocm_happy" != xno])

