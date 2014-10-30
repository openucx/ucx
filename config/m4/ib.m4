#
# Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
#
# $COPYRIGHT$
# $HEADER$
#

with_ib=no

#
# RC Support
#
AC_ARG_WITH([rc],
            [AC_HELP_STRING([--with-rc], [Compile with RC support])],
            [],
            [with_rc=yes])
AM_CONDITIONAL([HAVE_TL_RC], [test "x$with_rc" != xno])
AS_IF([test "x$with_rc" != xno], 
      [AC_DEFINE([HAVE_TL_RC], 1, [RC transport support])
       with_ib=yes
       transports="${transports},rc"])

AM_CONDITIONAL([HAVE_IB], [test "x$with_ib" != xno])
AS_IF([test "x$with_ib" != xno],
      [AC_DEFINE([HAVE_IB], 1, [IB support])])
