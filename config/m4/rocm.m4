#
# Copyright 2016 - 2017 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
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
        AC_CHECK_HEADERS([$with_rocm/hsa/include/hsa_ext_amd.h],
                       [AC_CHECK_DECLS([hsaKmtProcessVMRead,hsaKmtProcessVMWrite],
                           [rocm_happy="yes"],
                           [AC_MSG_WARN([ROCm without CMA support was detected. Disable.])
                            rocm_happy="no"],
                            [#include <$with_rocm/libhsakmt/include/libhsakmt/hsakmt.h>])
                           AS_IF([test "x$rocm_happy" == "xyes"],
                            [AC_DEFINE([HAVE_ROCM], 1, [Enable ROCm support])
                             transports="${transports},rocm"
                            AC_SUBST(ROCM_CPPFLAGS, "-I$with_rocm/hsa/include -I$with_rocm/libhsakmt/include/libhsakmt -DHAVE_ROCM=1")
                            AC_SUBST(ROCM_CFLAGS, "-I$with_rocm/hsa/include -I$with_rocm/libhsakmt/include/libhsakmt -DHAVE_ROCM=1")
                            AC_SUBST(ROCM_LDFLAGS, "-lhsa-runtime64 -L$with_rocm/hsa/lib")
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

