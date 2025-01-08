#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See file LICENSE for terms.
#

#
# EFA provider Support
#
AC_ARG_WITH([efa],
            [AS_HELP_STRING([--with-efa], [Enable EFA device support (default is guess)])],
            [], [with_efa=guess])

have_efa=no

AS_IF([test "x$with_ib" = "xno"], [with_efa=no])

AS_IF([test "x$with_efa" != xno],
      [
       save_LDFLAGS="$LDFLAGS"
       save_CFLAGS="$CFLAGS"
       save_CPPFLAGS="$CPPFLAGS"

       LDFLAGS="$IBVERBS_LDFLAGS $LDFLAGS"
       CFLAGS="$IBVERBS_CFLAGS $CFLAGS"
       CPPFLAGS="$IBVERBS_CPPFLAGS $CPPFLAGS"

       have_efa=yes
       AC_CHECK_HEADER([infiniband/efadv.h], [], [have_efa=no])
       AC_CHECK_LIB([efa], [efadv_query_device],
                    [AC_SUBST(EFA_LIB, [-lefa])],
                    [have_efa=no])

       AC_CHECK_DECLS([EFADV_DEVICE_ATTR_CAPS_RDMA_READ],
                      [], [], [#include <infiniband/efadv.h>])

       AS_IF([test "x$have_efa" = xyes],
             [
              uct_ib_modules="${uct_ib_modules}:efa"
             ],
             [AS_IF([test "x$with_efa" != xguess],
                    [AC_MSG_ERROR(EFA device support requested but libefa or EFA headers are not found)])])

       LDFLAGS="$save_LDFLAGS"
       CFLAGS="$save_CFLAGS"
       CPPFLAGS="$save_CPPFLAGS"
      ])


#
# For automake
#
AM_CONDITIONAL([HAVE_EFA],  [test "x$have_efa" = xyes])

AC_CONFIG_FILES([src/uct/ib/efa/Makefile src/uct/ib/efa/ucx-ib-efa.pc])
