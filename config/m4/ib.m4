#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
# $COPYRIGHT$
# $HEADER$
#

#
# Select IB transports
#
with_ib=no

#
# Check basic IB support: User wanted at least one IB transport, and we found
# verbs header file and library.
#
AS_IF([test "x$with_ib" != xno],
      [AC_CHECK_HEADER([infiniband/verbs.h], [],
                       [AC_MSG_WARN([ibverbs header files not found]);with_ib=no])
       save_LDFLAGS="$LDFLAGS"
       AC_CHECK_LIB([ibverbs], [ibv_get_device_list],
                    [AC_SUBST(IBVERBS_LDFLAGS, [-libverbs])],
                    [AC_MSG_WARN([libibverbs not found]);with_ib=no])
       LDFLAGS="$save_LDFLAGS"
      ])
AS_IF([test "x$with_ib" != xno],
      [AC_DEFINE([HAVE_IB], 1, [IB support])])
#
# RC Support
#
AC_ARG_WITH([rc],
            [AC_HELP_STRING([--with-rc], [Compile with IB Reliable Connection support])],
            [],
            [with_rc=yes])
AS_IF([test "x$with_rc" != xno -a "x$with_ib" !=  xno], 
      [AC_DEFINE([HAVE_TL_RC], 1, [RC transport support])
       transports="${transports},rc"])


AC_ARG_WITH([ud],
            [AC_HELP_STRING([--with-ud], [Compile with IB Unreliable Datagram support])],
            [],
            [with_ud=yes])
AS_IF([test "x$with_ud" != xno -a "x$with_ib" !=  xno],
      [AC_DEFINE([HAVE_TL_UD], 1, [UD transport support])
       transports="${transports},ud"])


AC_ARG_WITH([dc],
            [AC_HELP_STRING([--with-dc], [Compile with IB Dynamic Connection support])],
            [],
            [with_dc=yes])
AS_IF([test "x$with_dc" != xno -a "x$with_ib" !=  xno],
      [AC_CHECK_DECLS(IBV_EXP_QPT_DC_INI, [], [with_dc=no], [[#include <infiniband/verbs.h>]])
       AC_CHECK_MEMBERS([struct ibv_exp_dct_init_attr.inline_size], [] , [with_dc=no], [[#include <infiniband/verbs.h>]])
      ])
AS_IF([test "x$with_dc" != xno -a "x$with_ib" !=  xno],
      [AC_DEFINE([HAVE_TL_DC], 1, [DC transport support])
       transports="${transports},dc"])


#
# Check for experimental verbs support
#
AC_CHECK_HEADER([infiniband/verbs_exp.h],
                [AC_DEFINE([HAVE_VERBS_EXP_H], 1, [IB experimental verbs])
                 verbs_exp=yes],
                [verbs_exp=no])


#
# mlx5 PRM
#
with_mlx5_hw=no
AC_CHECK_HEADERS([infiniband/mlx5_hw.h],
                 [AC_MSG_NOTICE([Compiling with mlx5 bare-metal support])
                  AC_DEFINE([HAVE_MLX5_HW], 1, [mlx5 bare-metal support])
                  with_mlx5_hw=yes])


#
# For automake
#
AM_CONDITIONAL([HAVE_IB], [test "x$with_ib" != xno])
AM_CONDITIONAL([HAVE_TL_RC], [test "x$with_rc" != xno])
AM_CONDITIONAL([HAVE_MLX5_HW], [test "x$with_mlx5_hw" != xno])

mlnx_valg_libdir=/usr/lib64/mlnx_ofed/valgrind
AS_IF([test -d "$mlnx_valg_libdir"],
      [AC_MSG_NOTICE([Added $mlnx_valg_libdir to valgrind LD_LIBRARY_PATH])
       valgrind_libpath="$mlnx_valg_libdir:$valgrind_libpath"])
