#
# Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
#
# $COPYRIGHT$
# $HEADER$
#


#
# Select IB transports
#
with_ib=no


#
# RC Support
#
AC_ARG_WITH([rc],
            [AC_HELP_STRING([--with-rc], [Compile with IB Reliable Connection support])],
            [],
            [with_rc=yes])
AS_IF([test "x$with_rc" != xno], 
      [AC_DEFINE([HAVE_TL_RC], 1, [RC transport support])
       with_ib=yes
       transports="${transports},rc"])


AC_ARG_WITH([ud],
            [AC_HELP_STRING([--with-ud], [Compile with IB Unreliable Datagram support])],
            [],
            [with_ud=yes;with_ib=yes])
AS_IF([test "x$with_ud" != xno],
      [AC_DEFINE([HAVE_TL_UD], 1, [UD transport support])
       with_ib=yes
       transports="${transports},ud"])


AC_ARG_WITH([dc],
            [AC_HELP_STRING([--with-dc], [Compile with IB Dynamic Connection support])],
            [],
            [with_dc=yes;with_ib=yes])
AS_IF([test "x$with_dc" != xno],
      [AC_CHECK_DECLS(IBV_EXP_QPT_DC_INI, [], [with_dc=no], [[#include <infiniband/verbs.h>]])
       AC_CHECK_MEMBERS([struct ibv_exp_dct_init_attr.inline_size], [] , [with_dc=no], [[#include <infiniband/verbs.h>]])
      ])
AS_IF([test "x$with_dc" != xno],
      [AC_DEFINE([HAVE_TL_DC], 1, [DC transport support])
       with_ib=yes
       transports="${transports},dc"])


#
# Check basic IB support: User wanted at least one IB transport, and we found
# verbs header file and library.
#
AS_IF([test "x$with_ib" != xno],
      [AC_CHECK_HEADER([infiniband/verbs.h], [],
                       [AC_MSG_ERROR([ibverbs header files not found]);with_ib=no])
       save_LDFLAGS="$LDFLAGS"
       AC_CHECK_LIB([ibverbs], [ibv_get_device_list],
                    [AC_SUBST(IBVERBS_LDFLAGS, [-libverbs])],
                    [AC_MSG_ERROR([libibverbs not found]);with_ib=no])
       LDFLAGS="$save_LDFLAGS"
      ])
AS_IF([test "x$with_ib" != xno],
      [AC_DEFINE([HAVE_IB], 1, [IB support])])


#
# Check for experimental verbs support
#
AC_CHECK_HEADER([infiniband/verbs_exp.h],
                [AC_DEFINE([HAVE_VERBS_EXP_H], 1, [IB experimental verbs])
                 verbs_exp=yes],
                [verbs_exp=no])


#
# For automake
#
AM_CONDITIONAL([HAVE_IB], [test "x$with_ib" != xno])
AM_CONDITIONAL([HAVE_TL_RC], [test "x$with_rc" != xno])
