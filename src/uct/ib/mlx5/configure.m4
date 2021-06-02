#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# See file LICENSE for terms.
#

AC_ARG_WITH([mlx5],
           [AS_HELP_STRING([--with-mlx5=(DIR)], [Compile with mlx5 support.])],
           [], [with_mlx5=/usr])
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
# DEVX Support
#
AC_ARG_WITH([devx], [], [], [with_devx=check])


AS_IF([test "x$with_ib" = "xno"], [with_mlx5=no])

AS_IF([test "x$with_mlx5" = "xyes"], [with_mlx5=/usr])
AS_IF([test -d "$with_mlx5"],
      [str="with mlx5 support from $with_mlx5"],
      [with_mlx5=no; str="without mlx5 support"])

AS_IF([test "x$with_mlx5" != "xno"],
      [
       AS_IF([test -d "$with_mlx5/lib64"],[libsuff="64"],[libsuff=""])
       AS_IF([test "x$with_mlx5" = "x/usr"],
          [],
          [mlx5_incl_dir="-I$with_mlx5/include"
           mlx5_libs_dir="-L$with_mlx5/lib$libsuff"])

       save_LIBS="$LIBS"
       save_LDFLAGS="$LDFLAGS"
       save_CFLAGS="$CFLAGS"
       save_CPPFLAGS="$CPPFLAGS"

       AC_SUBST(MLX5_LDFLAGS,  ["$mlx5_libs_dir"])
       AC_SUBST(MLX5_CFLAGS,   ["$mlx5_incl_dir"])
       AC_SUBST(MLX5_CPPFLAGS, ["$mlx5_incl_dir"])

       LDFLAGS="$MLX5_LDFLAGS $LDFLAGS"
       CFLAGS="$MLX5_CFLAGS $CFLAGS"
       CPPFLAGS="$MLX5_CPPFLAGS $CPPFLAGS"

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
                                    [AC_SUBST(MLX5_LIBS, [-lmlx5-rdmav2])],[
              AC_CHECK_LIB([mlx5], [mlx5dv_query_device],
                                    [AC_SUBST(MLX5_LIBS, [-lmlx5])],
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

       AC_CHECK_MEMBERS([struct mlx5_wqe_av.base,
                         struct mlx5_grh_av.rmac],
                        [], [], [[#include <infiniband/$mlx5_include>]])

       AC_CHECK_MEMBERS([struct mlx5_cqe64.ib_stride_index],
                        [], [], [[#include <infiniband/$mlx5_include>]])

       AC_CHECK_DECLS([IBV_EXP_QPT_DC_INI],
                [have_dc_exp=yes], [], [[#include <infiniband/verbs.h>]])

       AS_IF([test "x$with_dc" != xno -a \( "x$have_dc_exp" = xyes -o "x$have_dc_dv" = xyes \) -a "x$with_mlx5_hw" = "xyes"], [
           AC_DEFINE([HAVE_TL_DC], 1, [DC transport support])
           AS_IF([test -n "$have_dc_dv"],
                 [AC_DEFINE([HAVE_DC_DV], 1, [DC DV support])], [
           AS_IF([test -n "$have_dc_exp"],
                 [AC_DEFINE([HAVE_DC_EXP], 1, [DC EXP support])])])],
           [with_dc=no])


       LIBS="$save_LIBS"
       LDFLAGS="$save_LDFLAGS"
       CFLAGS="$save_CFLAGS"
       CPPFLAGS="$save_CPPFLAGS"

       uct_ib_modules="${uct_ib_modules}:mlx5"
    ],
    [
        with_mlx5=no
        with_mlx5_hw=no
        with_mlx5_dv=no
        with_dc=no
    ])

#
# For automake
#
AM_CONDITIONAL([HAVE_MLX5],    [test "x$with_mlx5" != xno])
AM_CONDITIONAL([HAVE_TL_DC],   [test "x$with_dc" != xno])
AM_CONDITIONAL([HAVE_DC_DV],   [test -n "$have_dc_dv"])
AM_CONDITIONAL([HAVE_DC_EXP],  [test -n "$have_dc_exp"])
AM_CONDITIONAL([HAVE_MLX5_HW], [test "x$with_mlx5_hw" != xno])
AM_CONDITIONAL([HAVE_MLX5_DV], [test "x$with_mlx5_dv" = xyes])
AM_CONDITIONAL([HAVE_DEVX],    [test -n "$have_devx"])
AM_CONDITIONAL([HAVE_MLX5_HW_UD], [test "x$with_mlx5_hw" != xno -a "x$has_get_av" != xno])

AC_CONFIG_FILES([src/uct/ib/mlx5/Makefile])
