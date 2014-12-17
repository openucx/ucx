#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
# $COPYRIGHT$
# $HEADER$
#


AC_ARG_WITH([verbs],
        [AC_HELP_STRING([--with-verbs(=DIR)],
            [Build OpenFabrics support, adding DIR/include, DIR/lib, and DIR/lib64 to the search path for headers and libraries])],
        [],
        [with_verbs=/usr])

AS_IF([test "x$with_verbs" == "xyes"], [with_verbs=/usr])
AS_IF([test -d "$with_verbs"], [with_ib=yes; str="with verbs support from $with_verbs"], [with_ib=no; str="without verbs support"])

AC_MSG_NOTICE([Compiling $str])

#
# RC Support
#
AC_ARG_WITH([rc],
            [AC_HELP_STRING([--with-rc], [Compile with IB Reliable Connection support])],
            [],
            [with_rc=yes])



AC_ARG_WITH([ud],
            [AC_HELP_STRING([--with-ud], [Compile with IB Unreliable Datagram support])],
            [],
            [with_ud=yes])



AC_ARG_WITH([dc],
            [AC_HELP_STRING([--with-dc], [Compile with IB Dynamic Connection support])],
            [],
            [with_dc=yes])



#
# Check basic IB support: User wanted at least one IB transport, and we found
# verbs header file and library.
#
AS_IF([test "x$with_ib" == xyes],
        [
        save_LDFLAGS="$LDFLAGS"
        save_CFLAGS="$CFLAGS"
        save_CPPFLAGS="$CPPFLAGS"
        AC_CHECK_HEADER([infiniband/verbs.h], [], [AC_MSG_WARN([ibverbs header files not found]); with_ib=no])
        AC_CHECK_LIB([ibverbs], [ibv_get_device_list],
            [
            AC_SUBST(IBVERBS_LDFLAGS,  ["-L$with_verbs/lib64 -L$with_verbs/lib -libverbs"])
            AC_SUBST(IBVERBS_CPPFLAGS, [-I$with_verbs/include])
            AC_SUBST(IBVERBS_CFLAGS,   [-I$with_verbs/include])
            ],
            [AC_MSG_WARN([libibverbs not found]); with_ib=no])
        LDFLAGS="$save_LDFLAGS"
        CFLAGS="$save_CFLAGS"
        CPPFLAGS="$save_CPPFLAGS"
        ],[:])

AS_IF([test "x$with_ib" == xyes],
      [
       AC_CHECK_HEADER([infiniband/verbs_exp.h],
           [AC_DEFINE([HAVE_VERBS_EXP_H], 1, [IB experimental verbs])
           verbs_exp=yes],
           [verbs_exp=no])

       with_mlx5_hw=no
       AC_CHECK_HEADERS([infiniband/mlx5_hw.h],
           [AC_MSG_NOTICE([Compiling with mlx5 bare-metal support])
           AC_DEFINE([HAVE_MLX5_HW], 1, [mlx5 bare-metal support])
           with_mlx5_hw=yes])

       AC_CHECK_DECLS([IBV_LINK_LAYER_INFINIBAND], [], [], [[#include <infiniband/verbs.h>]])

       AC_DEFINE([HAVE_IB], 1, [IB support])

       AS_IF([test "x$with_dc" != xno],
           [AC_CHECK_DECLS(IBV_EXP_QPT_DC_INI, [], [with_dc=no], [[#include <infiniband/verbs.h>]])
           AC_CHECK_MEMBERS([struct ibv_exp_dct_init_attr.inline_size], [] , [with_dc=no], [[#include <infiniband/verbs.h>]])
           ])

       AS_IF([test "x$with_dc" != xno],
               [AC_DEFINE([HAVE_TL_DC], 1, [DC transport support])
               transports="${transports},dc"])

       AS_IF([test "x$with_rc" != xno], 
               [AC_DEFINE([HAVE_TL_RC], 1, [RC transport support])
               transports="${transports},rc"])

       AS_IF([test "x$with_ud" != xno],
               [AC_DEFINE([HAVE_TL_UD], 1, [UD transport support])
               transports="${transports},ud"])

       AS_IF([test -d "$with_verbs/lib64"],[libsuff="64"],[libsuff=""])
       mlnx_valg_libdir=$with_verbs/lib${libsuff}/mlnx_ofed/valgrind
       AC_MSG_NOTICE([Checking OFED valgrind libs $mlnx_valg_libdir])

       AS_IF([test -d "$mlnx_valg_libdir"],
               [AC_MSG_NOTICE([Added $mlnx_valg_libdir to valgrind LD_LIBRARY_PATH])
               valgrind_libpath="$mlnx_valg_libdir:$valgrind_libpath"])
])




#
# For automake
#
AM_CONDITIONAL([HAVE_IB], [test "x$with_ib" != xno])
AM_CONDITIONAL([HAVE_TL_RC], [test "x$with_rc" != xno])
AM_CONDITIONAL([HAVE_TL_DC], [test "x$with_dc" != xno])
AM_CONDITIONAL([HAVE_TL_UD], [test "x$with_ud" != xno])
AM_CONDITIONAL([HAVE_MLX5_HW], [test "x$with_mlx5_hw" != xno])

