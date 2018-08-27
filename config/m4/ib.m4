#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
# Copyright (C) The University of Tennessee and the University of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#


AC_ARG_WITH([verbs],
        [AC_HELP_STRING([--with-verbs(=DIR)],
            [Build OpenFabrics support, adding DIR/include, DIR/lib, and DIR/lib64 to the search path for headers and libraries])],
        [],
        [with_verbs=/usr])

AS_IF([test "x$with_verbs" == "xyes"], [with_verbs=/usr])
AS_IF([test -d "$with_verbs"], [with_ib=yes; str="with verbs support from $with_verbs"], [with_ib=no; str="without verbs support"])
AS_IF([test -d "$with_verbs/lib64"],[libsuff="64"],[libsuff=""])

AC_MSG_NOTICE([Compiling $str])

#
# RC Support
#
AC_ARG_WITH([rc],
            [AC_HELP_STRING([--with-rc], [Compile with IB Reliable Connection support])],
            [],
            [with_rc=yes])


#
# UD Support
#
AC_ARG_WITH([ud],
            [AC_HELP_STRING([--with-ud], [Compile with IB Unreliable Datagram support])],
            [],
            [with_ud=yes])


#
# DC Support
#
AC_ARG_WITH([dc],
            [AC_HELP_STRING([--with-dc], [Compile with IB Dynamic Connection support])],
            [],
            [with_dc=yes])


#
# CM (IB connection manager) Support
#
AC_ARG_WITH([cm],
            [AC_HELP_STRING([--with-cm], [Compile with IB Connection Manager support])],
            [],
            [with_cm=yes])


#
# mlx5 DV support
#
AC_ARG_WITH([mlx5-dv],
            [AC_HELP_STRING([--with-mlx5-dv], [Compile with mlx5 Direct Verbs
                support. Direct Verbs (DV) support provides additional
                acceleration capabilities that are not available in a
                regular mode.])],
            [],
            [with_mlx5_dv=yes])


#
# TM (IB Tag Matching) Support
#
AC_ARG_WITH([ib-hw-tm],
            [AC_HELP_STRING([--with-ib-hw-tm], [Compile with IB Tag Matching support])],
            [],
            [with_ib_hw_tm=yes])


#
# DM Support
#
AC_ARG_WITH([dm],
            [AC_HELP_STRING([--with-dm], [Compile with Device Memory support])],
            [],
            [with_dm=yes])


#
# Check basic IB support: User wanted at least one IB transport, and we found
# verbs header file and library.
#
AS_IF([test "x$with_ib" == xyes],
        [
        save_LDFLAGS="$LDFLAGS"
        save_CFLAGS="$CFLAGS"
        save_CPPFLAGS="$CPPFLAGS"
        AS_IF([test x/usr == "x$with_verbs"],
          [],
          [verbs_incl="-I$with_verbs/include"
           verbs_libs="-L$with_verbs/lib$libsuff"])
        LDFLAGS="$verbs_libs $LDFLAGS"
        CFLAGS="$verbs_incl $CFLAGS"
        CPPFLAGS="$verbs_incl $CPPFLAGS"
        AC_CHECK_HEADER([infiniband/verbs.h], [],
                        [AC_MSG_WARN([ibverbs header files not found]); with_ib=no])
        AC_CHECK_LIB([ibverbs], [ibv_get_device_list],
            [
            AC_SUBST(IBVERBS_LDFLAGS,  ["$verbs_libs -libverbs"])
            AC_SUBST(IBVERBS_DIR,      ["$with_verbs"])
            AC_SUBST(IBVERBS_CPPFLAGS, ["$verbs_incl"])
            AC_SUBST(IBVERBS_CFLAGS,   ["$verbs_incl"])
            ],
            [AC_MSG_WARN([libibverbs not found]); with_ib=no])

        have_ib_funcs=yes
        LDFLAGS="$LDFLAGS $IBVERBS_LDFLAGS"
        AC_CHECK_DECLS([ibv_wc_status_str, \
                        ibv_event_type_str, \
                        ibv_query_gid, \
                        ibv_get_device_name, \
                        ibv_create_srq, \
                        ibv_get_async_event],
                       [],
                       [have_ib_funcs=no],
                       [#include <infiniband/verbs.h>])
        AS_IF([test "x$have_ib_funcs" != xyes],
              [AC_MSG_WARN([Some IB verbs are not found. Please make sure OFED version is 1.5 or above.])
               with_ib=no])

        LDFLAGS="$save_LDFLAGS"
        CFLAGS="$save_CFLAGS"
        CPPFLAGS="$save_CPPFLAGS"
        ],[:])

AS_IF([test "x$with_ib" == xyes],
      [
       save_LDFLAGS="$LDFLAGS"
       save_CFLAGS="$CFLAGS"
       save_CPPFLAGS="$CPPFLAGS"
       LDFLAGS="$IBVERBS_LDFAGS $LDFLAGS"
       CFLAGS="$IBVERBS_CFLAGS $CFLAGS"
       CPPFLAGS="$IBVERBS_CPPFLAGS $CPPFLAGS"
       AC_CHECK_HEADER([infiniband/verbs_exp.h],
           [AC_DEFINE([HAVE_VERBS_EXP_H], 1, [IB experimental verbs])
           verbs_exp=yes],
           [verbs_exp=no])

       AC_CHECK_MEMBERS([struct ibv_exp_device_attr.exp_device_cap_flags,
                         struct ibv_exp_device_attr.odp_caps,
                         struct ibv_exp_device_attr.odp_caps.per_transport_caps.dc_odp_caps,
                         struct ibv_exp_device_attr.odp_mr_max_size,
                         struct ibv_exp_qp_init_attr.max_inl_recv,
                         struct ibv_async_event.element.dct],
                        [], [], [[#include <infiniband/verbs_exp.h>]])

       AC_CHECK_DECLS([IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN],
                      [have_cq_io=yes], [], [[#include <infiniband/verbs.h>]])

       AC_CHECK_DECLS([IBV_EXP_CQ_IGNORE_OVERRUN],
                      [have_cq_io=yes], [], [[#include <infiniband/verbs_exp.h>]])

       AS_IF([test "x$with_mlx5_dv" != xno], [
               AC_MSG_NOTICE([Checking for legacy bare-metal support])
               AC_CHECK_HEADERS([infiniband/mlx5_hw.h],
                               [with_mlx5_hw=yes
                                mlx5_include=mlx5_hw.h
                       AC_CHECK_DECLS([
                           ibv_mlx5_exp_get_qp_info,
                           ibv_mlx5_exp_get_cq_info,
                           ibv_mlx5_exp_get_srq_info,
                           ibv_mlx5_exp_update_cq_ci,
                           MLX5_WQE_CTRL_SOLICITED],
                                  [], [], [[#include <infiniband/mlx5_hw.h>]])
                       AC_CHECK_MEMBERS([struct mlx5_srq.cmd_qp],
                                  [], [with_ib_hw_tm=no],
                                      [[#include <infiniband/mlx5_hw.h>]])
                       AC_CHECK_MEMBERS([struct mlx5_ah.ibv_ah],
                                  [has_get_av=yes], [],
                                      [[#include <infiniband/mlx5_hw.h>]])
                       AC_CHECK_MEMBERS([struct ibv_mlx5_qp_info.bf.need_lock],
                               [],
                               [AC_MSG_WARN([Cannot use mlx5 QP because it assumes dedicated BF])
                                AC_MSG_WARN([Please upgrade MellanoxOFED to 3.0 or above])
                                with_mlx5_hw=no],
                               [[#include <infiniband/mlx5_hw.h>]])
                       AC_CHECK_DECLS([
                           IBV_EXP_QP_INIT_ATTR_RES_DOMAIN,
                           IBV_EXP_RES_DOMAIN_THREAD_MODEL,
                           ibv_exp_create_res_domain,
                           ibv_exp_destroy_res_domain],
                                [AC_DEFINE([HAVE_IBV_EXP_RES_DOMAIN], 1, [IB resource domain])
                                 has_res_domain=yes], [], [[#include <infiniband/verbs_exp.h>]])
                               ], [with_mlx5_hw=no])

              AC_MSG_NOTICE([Checking for DV bare-metal support])
              AC_CHECK_HEADERS([infiniband/mlx5dv.h],
                               [with_mlx5_hw=yes
                                with_mlx5_dv=yes
                                mlx5_include=mlx5dv.h
                       AC_CHECK_LIB([mlx5-rdmav2], [mlx5dv_query_device],
                                    [AC_SUBST(LIB_MLX5, [-lmlx5-rdmav2])],[
                       AC_CHECK_LIB([mlx5], [mlx5dv_query_device],
                                    [AC_SUBST(LIB_MLX5, [-lmlx5])],
                                    [with_mlx5_dv=no], [-libverbs])], [-libverbs])])

              AS_IF([test "x$have_cq_io" == xyes ], [
                       AC_CHECK_DECLS([
                           mlx5dv_init_obj],
                                  [], [], [[#include <infiniband/mlx5dv.h>]])
                       AC_CHECK_MEMBERS([struct mlx5dv_cq.cq_uar],
                                  [], [], [[#include <infiniband/mlx5dv.h>]])
                       AC_CHECK_DECLS([MLX5DV_OBJ_AH], [has_get_av=yes],
                                      [], [[#include <infiniband/mlx5dv.h>]])
                       AC_CHECK_DECLS([ibv_alloc_td],
                                  [has_res_domain=yes], [], [[#include <infiniband/verbs.h>]])])])

       AS_IF([test "x$has_res_domain" == xyes], [], [
               AC_MSG_WARN([Cannot use mlx5 accel because resource domains are not supported])
               AC_MSG_WARN([Please upgrade MellanoxOFED to 3.1 or above])
               with_mlx5_hw=no])

       AS_IF([test "x$with_mlx5_hw" == xyes],
             [AC_MSG_NOTICE([Compiling with mlx5 bare-metal support])
              AC_DEFINE([HAVE_MLX5_HW], 1, [mlx5 bare-metal support])
              AS_IF([test "x$has_get_av" == xyes],
                 [AC_DEFINE([HAVE_MLX5_HW_UD], 1, [mlx5 UD bare-metal support])], [])], [])

       AC_CHECK_DECLS([IBV_LINK_LAYER_INFINIBAND,
                       IBV_LINK_LAYER_ETHERNET,
                       IBV_EVENT_GID_CHANGE],
                      [], [], [[#include <infiniband/verbs.h>]])

       AC_CHECK_DECLS([IBV_EXP_ACCESS_ALLOCATE_MR,
                       IBV_EXP_ACCESS_ON_DEMAND,
                       IBV_EXP_DEVICE_MR_ALLOCATE,
                       IBV_EXP_WR_NOP,
                       IBV_EXP_DEVICE_DC_TRANSPORT,
                       IBV_EXP_ATOMIC_HCA_REPLY_BE,
                       IBV_EXP_PREFETCH_WRITE_ACCESS,
                       IBV_EXP_QP_OOO_RW_DATA_PLACEMENT,
                       IBV_EXP_DCT_OOO_RW_DATA_PLACEMENT,
                       IBV_EXP_CQ_MODERATION,
                       ibv_exp_reg_mr,
                       ibv_exp_create_qp,
                       ibv_exp_prefetch_mr,
                       ibv_exp_create_srq,
                       ibv_exp_setenv,
                       ibv_exp_query_gid_attr],
                      [], [], [[#include <infiniband/verbs_exp.h>]])

       AC_CHECK_DECLS([ibv_exp_post_send,
                       IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP,
                       IBV_EXP_WR_EXT_MASKED_ATOMIC_FETCH_AND_ADD,
                       IBV_EXP_QP_INIT_ATTR_ATOMICS_ARG,
                       IBV_EXP_SEND_EXT_ATOMIC_INLINE],
                      [],
                      [have_ext_atomics=no],
                      [[#include <infiniband/verbs_exp.h>]])

       AC_CHECK_DECLS(IBV_EXP_DEVICE_ATTR_RESERVED_2, [], [],
                      [[#include <infiniband/verbs_exp.h>]])

       # UMR support
       AC_CHECK_DECLS(IBV_EXP_MR_INDIRECT_KLMS,
                     [AC_DEFINE([HAVE_EXP_UMR], 1, [IB UMR support])],
                     [],
                     [[#include <infiniband/verbs.h>]])

       AC_CHECK_DECLS(IBV_EXP_QP_CREATE_UMR,
                     [AC_DEFINE([HAVE_IBV_EXP_QP_CREATE_UMR], 1, [IB QP Create UMR support])],
                     [],
                     [[#include <infiniband/verbs.h>]])

       AC_CHECK_MEMBERS([struct ibv_exp_qp_init_attr.umr_caps],
                        [AC_DEFINE([HAVE_IBV_EXP_QP_CREATE_UMR_CAPS], 1, [Support UMR max caps v2])],
                        [],
                        [[#include <infiniband/verbs.h>]])

       AC_CHECK_DECLS(IBV_EXP_MR_FIXED_BUFFER_SIZE,
                     [AC_DEFINE([HAVE_EXP_UMR_KSM], 1, [IB UMR KSM support])],
                     [],
                     [[#include <infiniband/verbs.h>]])

       # Extended atomics
       AS_IF([test "x$have_ext_atomics" != xno],
             [AC_DEFINE([HAVE_IB_EXT_ATOMICS], 1, [IB extended atomics support])],
             [AC_MSG_WARN([Compiling without extended atomics support])])

       # Check for driver which exposes masked atomics endianity per size
       AC_CHECK_MEMBER(struct ibv_exp_masked_atomic_params.masked_log_atomic_arg_sizes_network_endianness,
                       [AC_DEFINE([HAVE_MASKED_ATOMICS_ENDIANNESS], 1, [have masked atomic endianness])],
                       [], [[#include <infiniband/verbs_exp.h>]])

       AC_CHECK_DECLS(IBV_EXP_ODP_SUPPORT_IMPLICIT, [], [],
                      [[#include <infiniband/verbs.h>]])

       AC_CHECK_MEMBERS([struct mlx5_wqe_av.base,
                         struct mlx5_grh_av.rmac],
                        [], [], [[#include <infiniband/$mlx5_include>]])

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

       AS_IF([test "x$with_cm" != xno],
           [save_LIBS="$LIBS"
            AC_CHECK_LIB([ibcm], [ib_cm_send_req],
                         [AC_SUBST(IBCM_LIBS, [-libcm])],
                         [with_cm=no])
            LIBS="$save_LIBS"
           ])
       AS_IF([test "x$with_cm" != xno],
           [AC_DEFINE([HAVE_TL_CM], 1, [Connection manager support])
           transports="${transports},cm"])

       # XRQ with Tag Matching support
       AS_IF([test "x$with_ib_hw_tm" != xno],
           [AC_CHECK_MEMBER([struct ibv_exp_tmh.tag], [], [with_ib_hw_tm=no],
                            [[#include <infiniband/verbs_exp.h>]])
           ])
       AS_IF([test "x$with_ib_hw_tm" != xno],
           [AC_DEFINE([IBV_EXP_HW_TM], 1, [IB Tag Matching support])
            AC_CHECK_MEMBERS([struct ibv_exp_create_srq_attr.dc_offload_params],
                             [AC_DEFINE([IBV_EXP_HW_TM_DC], 1, [DC Tag Matching support])],
                             [], [#include <infiniband/verbs_exp.h>])
           ])

       # Device Memory support
       AS_IF([test "x$with_dm" != xno],
           [AC_TRY_COMPILE([#include <infiniband/verbs_exp.h>],
               [
                   struct ibv_exp_dm ibv_dm;
                   struct ibv_exp_alloc_dm_attr dm_attr;
                   void* a1 = ibv_exp_alloc_dm;
                   void* a2 = ibv_exp_reg_mr;
                   void* a3 = ibv_dereg_mr;
                   void* a4 = ibv_exp_free_dm;
               ],
               [AC_DEFINE([HAVE_IBV_EXP_DM], 1, [Device Memory support])],
               [])
           ])

       AC_CHECK_DECLS([ibv_cmd_modify_qp],
                      [], [], [[#include <infiniband/driver.h>]])

       mlnx_valg_libdir=$with_verbs/lib${libsuff}/mlnx_ofed/valgrind
       AC_MSG_NOTICE([Checking OFED valgrind libs $mlnx_valg_libdir])

       AS_IF([test -d "$mlnx_valg_libdir"],
               [AC_MSG_NOTICE([Added $mlnx_valg_libdir to valgrind LD_LIBRARY_PATH])
               valgrind_libpath="$mlnx_valg_libdir:$valgrind_libpath"])
       LDFLAGS="$save_LDFLAGS"
       CFLAGS="$save_CFLAGS"
       CPPFLAGS="$save_CPPFLAGS"
    ],
    [
        with_dc=no
        with_rc=no
        with_ud=no
        with_mlx5_hw=no
        with_mlx5_dv=no
        with_ib_hw_tm=no
    ])


#
# For automake
#
AM_CONDITIONAL([HAVE_IB],      [test "x$with_ib" != xno])
AM_CONDITIONAL([HAVE_TL_RC],   [test "x$with_rc" != xno])
AM_CONDITIONAL([HAVE_TL_DC],   [test "x$with_dc" != xno])
AM_CONDITIONAL([HAVE_TL_UD],   [test "x$with_ud" != xno])
AM_CONDITIONAL([HAVE_TL_CM],   [test "x$with_cm" != xno])
AM_CONDITIONAL([HAVE_MLX5_HW], [test "x$with_mlx5_hw" != xno])
AM_CONDITIONAL([HAVE_MLX5_DV], [test "x$with_mlx5_dv" != xno])
AM_CONDITIONAL([HAVE_MLX5_HW_UD], [test "x$with_mlx5_hw" != xno -a "x$has_get_av" != xno])
AM_CONDITIONAL([HAVE_IBV_EX_HW_TM], [test "x$with_ib_hw_tm"  != xno])
