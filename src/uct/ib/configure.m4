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

AS_IF([test "x$with_verbs" = "xyes"], [with_verbs=/usr])
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
# mlx5 DV support
#
AC_ARG_WITH([mlx5-dv],
            [AC_HELP_STRING([--with-mlx5-dv], [Compile with mlx5 Direct Verbs
                support. Direct Verbs (DV) support provides additional
                acceleration capabilities that are not available in a
                regular mode.])])

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
# DEVX Support
#
AC_ARG_WITH([devx], [], [], [with_devx=check])

#
# Check basic IB support: User wanted at least one IB transport, and we found
# verbs header file and library.
#
AS_IF([test "x$with_ib" = "xyes"],
        [
        save_LDFLAGS="$LDFLAGS"
        save_CFLAGS="$CFLAGS"
        save_CPPFLAGS="$CPPFLAGS"
        AS_IF([test "x/usr" = "x$with_verbs"],
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
        AC_CHECK_DECLS([ibv_wc_status_str,
                        ibv_event_type_str,
                        ibv_query_gid,
                        ibv_get_device_name,
                        ibv_create_srq,
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

AS_IF([test "x$with_ib" = "xyes"],
      [
       save_LDFLAGS="$LDFLAGS"
       save_CFLAGS="$CFLAGS"
       save_CPPFLAGS="$CPPFLAGS"
       LDFLAGS="$IBVERBS_LDFLAGS $LDFLAGS"
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

              AC_CHECK_LIB([mlx5-rdmav2], [mlx5dv_query_device],
                                    [AC_SUBST(LIB_MLX5, [-lmlx5-rdmav2])],[
              AC_CHECK_LIB([mlx5], [mlx5dv_query_device],
                                    [AC_SUBST(LIB_MLX5, [-lmlx5])],
                                    [with_mlx5_dv=no], [-libverbs])], [-libverbs])

              AS_IF([test "x$with_mlx5_dv" != xno], [
                       AC_CHECK_HEADERS([infiniband/mlx5dv.h],
                               [with_mlx5_hw=yes
                                with_mlx5_dv=yes
                                mlx5_include=mlx5dv.h], [], [ ])])

              AS_IF([test "x$with_mlx5_dv" = "xyes" -a "x$have_cq_io" = "xyes" ], [
                       AC_CHECK_DECLS([
                           mlx5dv_init_obj,
                           mlx5dv_create_qp,
                           mlx5dv_is_supported,
                           mlx5dv_devx_subscribe_devx_event,
                           MLX5DV_CQ_INIT_ATTR_MASK_CQE_SIZE,
                           MLX5DV_QP_CREATE_ALLOW_SCATTER_TO_CQE,
                           MLX5DV_UAR_ALLOC_TYPE_BF,
                           MLX5DV_UAR_ALLOC_TYPE_NC],
                                  [], [], [[#include <infiniband/mlx5dv.h>]])
                       AC_CHECK_MEMBERS([struct mlx5dv_cq.cq_uar],
                                  [], [], [[#include <infiniband/mlx5dv.h>]])
                       AC_CHECK_DECLS([MLX5DV_OBJ_AH], [has_get_av=yes],
                                      [], [[#include <infiniband/mlx5dv.h>]])
                       AC_CHECK_DECLS([MLX5DV_DCTYPE_DCT],
                                  [have_dc_dv=yes], [], [[#include <infiniband/mlx5dv.h>]])
                       AC_CHECK_DECLS([ibv_alloc_td],
                                  [has_res_domain=yes], [], [[#include <infiniband/verbs.h>]])])

              AC_CHECK_DECLS([ibv_alloc_td],
                      [has_res_domain=yes], [], [[#include <infiniband/verbs.h>]])])

       AS_IF([test "x$with_devx" != xno], [
            AC_CHECK_DECL(MLX5DV_CONTEXT_FLAGS_DEVX, [
                 AC_DEFINE([HAVE_DEVX], [1], [DEVX support])
                 have_devx=yes
            ], [
                 AS_IF([test "x$with_devx" != xcheck],
                       [AC_MSG_ERROR([devx requested but not found])])
            ], [[#include <infiniband/mlx5dv.h>]])])

       AS_IF([test "x$has_res_domain" = "xyes" -a "x$have_cq_io" = "xyes" ], [], [
               with_mlx5_hw=no])

       AS_IF([test "x$with_mlx5_hw" = "xyes"],
             [AC_MSG_NOTICE([Compiling with mlx5 bare-metal support])
              AC_DEFINE([HAVE_MLX5_HW], 1, [mlx5 bare-metal support])
              AS_IF([test "x$has_get_av" = "xyes"],
                 [AC_DEFINE([HAVE_MLX5_HW_UD], 1, [mlx5 UD bare-metal support])])])

       AC_CHECK_DECLS([IBV_LINK_LAYER_INFINIBAND,
                       IBV_LINK_LAYER_ETHERNET,
                       IBV_EVENT_GID_CHANGE,
                       ibv_create_qp_ex,
                       ibv_create_srq_ex],
                      [], [], [[#include <infiniband/verbs.h>]])

       # We shouldn't confuse upstream ibv_query_device_ex with
       # legacy MOFED one, distinguish by arguments number
       AC_CHECK_DECL(ibv_query_device_ex, [
       AC_TRY_COMPILE([#include <infiniband/verbs.h>],
                      [ibv_query_device_ex(NULL, NULL, NULL)],
                      [AC_DEFINE([HAVE_DECL_IBV_QUERY_DEVICE_EX], 1,
                          [have upstream ibv_query_device_ex])])],
                          [], [[#include <infiniband/verbs.h>]])

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
                       IBV_EXP_DEVICE_ATTR_PCI_ATOMIC_CAPS,
                       ibv_exp_reg_mr,
                       ibv_exp_create_qp,
                       ibv_exp_prefetch_mr,
                       ibv_exp_create_srq,
                       ibv_exp_setenv,
                       ibv_exp_query_gid_attr,
                       ibv_exp_query_device],
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

       AC_CHECK_MEMBERS([struct ibv_device_attr_ex.pci_atomic_caps],
                        [], [], [[#include <infiniband/verbs.h>]])

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

       AC_CHECK_DECLS(IBV_EXP_ACCESS_ON_DEMAND, [with_odp=yes], [],
                      [[#include <infiniband/verbs_exp.h>]])

       AC_CHECK_DECLS(IBV_ACCESS_ON_DEMAND, [with_odp=yes], [],
                      [[#include <infiniband/verbs.h>]])

       AS_IF([test "x$with_odp" = "xyes" ], [
           AC_DEFINE([HAVE_ODP], 1, [ODP support])

           AC_CHECK_DECLS(IBV_EXP_ODP_SUPPORT_IMPLICIT, [with_odp_i=yes], [],
                          [[#include <infiniband/verbs_exp.h>]])

           AC_CHECK_DECLS(IBV_ODP_SUPPORT_IMPLICIT, [with_odp_i=yes], [],
                          [[#include <infiniband/verbs.h>]])

           AS_IF([test "x$with_odp_i" = "xyes" ], [
               AC_DEFINE([HAVE_ODP_IMPLICIT], 1, [Implicit ODP support])])])

       AC_CHECK_DECLS([IBV_ACCESS_RELAXED_ORDERING,
                       IBV_QPF_GRH_REQUIRED],
                      [], [], [[#include <infiniband/verbs.h>]])

       AC_CHECK_DECLS(ibv_exp_prefetch_mr, [with_prefetch=yes], [],
                      [[#include <infiniband/verbs_exp.h>]])

       AC_CHECK_DECLS(ibv_advise_mr, [with_prefetch=yes], [],
                      [[#include <infiniband/verbs.h>]])

       AS_IF([test "x$with_prefetch" = "xyes" ], [
           AC_DEFINE([HAVE_PREFETCH], 1, [Prefetch support])])

       AC_CHECK_MEMBERS([struct mlx5_wqe_av.base,
                         struct mlx5_grh_av.rmac],
                        [], [], [[#include <infiniband/$mlx5_include>]])

       AC_CHECK_MEMBERS([struct mlx5_cqe64.ib_stride_index],
                        [], [], [[#include <infiniband/$mlx5_include>]])

       AC_DEFINE([HAVE_IB], 1, [IB support])

       AC_CHECK_DECLS([IBV_EXP_QPT_DC_INI],
                [have_dc_exp=yes], [], [[#include <infiniband/verbs.h>]])

       AS_IF([test "x$with_dc" != xno -a \( "x$have_dc_exp" = xyes -o "x$have_dc_dv" = xyes \) -a "x$with_mlx5_hw" = "xyes"], [
           AC_DEFINE([HAVE_TL_DC], 1, [DC transport support])
           AS_IF([test -n "$have_dc_dv"],
                 [AC_DEFINE([HAVE_DC_DV], 1, [DC DV support])], [
           AS_IF([test -n "$have_dc_exp"],
                 [AC_DEFINE([HAVE_DC_EXP], 1, [DC EXP support])])])],
           [with_dc=no])

       AS_IF([test "x$with_rc" != xno],
           [AC_DEFINE([HAVE_TL_RC], 1, [RC transport support])])

       AS_IF([test "x$with_ud" != xno],
           [AC_DEFINE([HAVE_TL_UD], 1, [UD transport support])])

       # XRQ with Tag Matching support
       AS_IF([test "x$with_ib_hw_tm" != xno],
           [AC_CHECK_HEADERS([infiniband/tm_types.h])
            AC_CHECK_MEMBER([struct ibv_exp_tmh.tag], [with_ib_hw_tm=exp], [],
                            [[#include <infiniband/verbs_exp.h>]])
            AC_CHECK_MEMBER([struct ibv_tmh.tag], [with_ib_hw_tm=upstream], [],
                            [[#include <infiniband/tm_types.h>]])
           ])
       AS_IF([test "x$with_ib_hw_tm" = xexp],
           [AC_CHECK_MEMBERS([struct ibv_exp_create_srq_attr.dc_offload_params], [
            AC_DEFINE([IBV_HW_TM], 1, [IB Tag Matching support])
                      ], [], [#include <infiniband/verbs_exp.h>])
           ])
       AS_IF([test "x$with_ib_hw_tm" = xupstream],
           [AC_DEFINE([IBV_HW_TM], 1, [IB Tag Matching support])
            AC_CHECK_MEMBERS([struct ibv_tm_caps.flags], [], [],
                             [#include <infiniband/verbs.h>])])

       # Device Memory support
       AS_IF([test "x$with_dm" != xno], [
           AC_CHECK_DECLS([ibv_exp_alloc_dm],
               [AC_DEFINE([HAVE_IBV_DM], 1, [Device Memory support])
                AC_DEFINE([HAVE_IBV_EXP_DM], 1, [Device Memory support (EXP)])],
               [], [[#include <infiniband/verbs_exp.h>]])
           AC_CHECK_DECLS([ibv_alloc_dm],
               [AC_DEFINE([HAVE_IBV_DM], 1, [Device Memory support])],
               [], [[#include <infiniband/verbs.h>]])])

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

       uct_modules="${uct_modules}:ib"
    ],
    [
        with_dc=no
        with_rc=no
        with_ud=no
        with_mlx5_hw=no
        with_mlx5_dv=no
    ])

#
# For automake
#
AM_CONDITIONAL([HAVE_IB],      [test "x$with_ib" != xno])
AM_CONDITIONAL([HAVE_TL_RC],   [test "x$with_rc" != xno])
AM_CONDITIONAL([HAVE_TL_DC],   [test "x$with_dc" != xno])
AM_CONDITIONAL([HAVE_DC_DV],   [test -n "$have_dc_dv"])
AM_CONDITIONAL([HAVE_DC_EXP],  [test -n "$have_dc_exp"])
AM_CONDITIONAL([HAVE_TL_UD],   [test "x$with_ud" != xno])
AM_CONDITIONAL([HAVE_MLX5_HW], [test "x$with_mlx5_hw" != xno])
AM_CONDITIONAL([HAVE_MLX5_DV], [test "x$with_mlx5_dv" = xyes])
AM_CONDITIONAL([HAVE_DEVX],    [test -n "$have_devx"])
AM_CONDITIONAL([HAVE_EXP],     [test "x$verbs_exp" != xno])
AM_CONDITIONAL([HAVE_MLX5_HW_UD], [test "x$with_mlx5_hw" != xno -a "x$has_get_av" != xno])

uct_ib_modules=""
m4_include([src/uct/ib/cm/configure.m4])
m4_include([src/uct/ib/rdmacm/configure.m4])
AC_DEFINE_UNQUOTED([uct_ib_MODULES], ["${uct_ib_modules}"], [IB loadable modules])
AC_CONFIG_FILES([src/uct/ib/Makefile])
