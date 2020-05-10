#
# Copyright (C) Huawei Technologies Co., Ltd. 2019.  ALL RIGHTS RESERVED.
# Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

#
# Enable UCG - Group collective operations component
#

ucg_modules=""

m4_sinclude([src/ucg/configure.m4]) # m4_sinclude() silently ignores errors

AC_ARG_ENABLE([ucg],
              AS_HELP_STRING([--enable-ucg],
                             [Enable the group collective operations (experimental component), default: NO]),
              [],
              [enable_ucg=no])
AS_IF([test "x$enable_ucg" != xno],
      [ucg_modules=":builtin"
       AC_DEFINE([ENABLE_UCG], [1],
                 [Enable Groups and collective operations support (UCG)])
       AC_MSG_NOTICE([Building with Groups and collective operations support (UCG)])
      ])
AS_IF([test -f ${ac_confdir}/src/ucg/Makefile.am],
      [AC_SUBST([UCG_SUBDIR], [src/ucg])])

AM_CONDITIONAL([HAVE_UCG], [test "x$enable_ucg" != xno])
