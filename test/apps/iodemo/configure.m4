#
# Copyright (c) NVIDIA CORPORATION. 2021. All rights reserved.
# See file LICENSE for terms.
#

#
# io_demo CUDA support
#
AC_ARG_WITH([iodemo_cuda],
            [AC_HELP_STRING([--with-iodemo-cuda], [Build io_demo example with CUDA support])],
            [],
            [with_iodemo_cuda=no])

AS_IF([test "x$with_iodemo_cuda" != xno],
      [AC_DEFINE([WITH_IODEMO_CUDA], 1, [io_demo CUDA support])])

#
# For automake
#
AM_CONDITIONAL([IODEMO_CUDA],   [test "x$with_iodemo_cuda" != xno])
