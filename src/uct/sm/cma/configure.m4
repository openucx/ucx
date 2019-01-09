#
# Copyright (C) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#
cma_happy="no"
AC_ARG_ENABLE([cma],
              [AC_HELP_STRING([--enable-cma],
                              [Enable Cross Memory Attach])],
                              [],
                              [enable_cma=yes])

AS_IF([test "x$enable_cma" != xno],
      [AC_CHECK_HEADERS([sys/uio.h],
            [AC_CHECK_FUNC(process_vm_readv,
                           [cma_happy="yes"],
                           [cma_happy="no"])
             AS_IF([test "x$cma_happy" == xyes],
             [AC_DEFINE([HAVE_CMA], 1, [CMA support])
             transports="${transports},cma"],
             [])], [])]
[])

AM_CONDITIONAL([HAVE_CMA], [test "x$cma_happy" != xno])
