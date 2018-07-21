/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#if HAVE_INFINIBAND_MLX5_HW_H

#include "ib_mlx5_hw.h"

#include <ucs/debug/log.h>
#include <ucs/debug/assert.h>
#include <ucs/arch/bitops.h>
#include <uct/ib/base/ib_verbs.h>
#include <infiniband/mlx5_hw.h>
#include <string.h>

/* Since this file intended to emulate DV using legacy mlx5_hw.h definitions
 * we include DV declarations. */
#define UCT_IB_MLX5_H_
#include "ib_mlx5_dv.h"

static void UCS_F_MAYBE_UNUSED uct_ib_mlx5_obj_error(const char *obj_name)
{
    ucs_error("Failed to get mlx5 %s information. Please make sure the installed "
              "libmlx5 version matches the one UCX was compiled with (%s)",
              obj_name, LIB_MLX5_VER);
}

#if !HAVE_DECL_MLX5DV_INIT_OBJ
ucs_status_t uct_ib_mlx5_get_qp_info(struct ibv_qp *qp, uct_ib_mlx5dv_qp_t *qp_info)
{
#if HAVE_DECL_IBV_MLX5_EXP_GET_QP_INFO
    struct ibv_mlx5_qp_info ibv_qp_info;
    int ret;

    ret = ibv_mlx5_exp_get_qp_info(qp, &ibv_qp_info);
    if (ret != 0) {
        uct_ib_mlx5_obj_error("qp");
        return UCS_ERR_NO_DEVICE;
    }

    qp_info->dv.dbrec      = ibv_qp_info.dbrec;
    qp_info->dv.sq.buf     = ibv_qp_info.sq.buf;
    qp_info->dv.sq.wqe_cnt = ibv_qp_info.sq.wqe_cnt;
    qp_info->dv.sq.stride  = ibv_qp_info.sq.stride;
    qp_info->dv.rq.buf     = ibv_qp_info.rq.buf;
    qp_info->dv.rq.wqe_cnt = ibv_qp_info.rq.wqe_cnt;
    qp_info->dv.rq.stride  = ibv_qp_info.rq.stride;
    qp_info->dv.bf.reg     = ibv_qp_info.bf.reg;
    qp_info->dv.bf.size    = ibv_qp_info.bf.size;
#else
    struct mlx5_qp *mqp = ucs_container_of(qp, struct mlx5_qp, verbs_qp.qp);

    if ((mqp->sq.cur_post != 0) || (mqp->rq.head != 0)) {
        ucs_warn("cur_post=%d head=%d need_lock=%d", mqp->sq.cur_post,
                 mqp->rq.head, mqp->bf->need_lock);
        return UCS_ERR_NO_DEVICE;
    }

    qp_info->dv.qpn        = qp->qp_num;
    qp_info->dv.dbrec      = mqp->db;
    qp_info->dv.sq.buf     = mqp->buf.buf + mqp->sq.offset;
    qp_info->dv.sq.wqe_cnt = mqp->sq.wqe_cnt;
    qp_info->dv.sq.stride  = 1 << mqp->sq.wqe_shift;
    qp_info->dv.rq.buf     = mqp->buf.buf + mqp->rq.offset;
    qp_info->dv.rq.wqe_cnt = mqp->rq.wqe_cnt;
    qp_info->dv.rq.stride  = 1 << mqp->rq.wqe_shift;
    qp_info->dv.bf.reg     = mqp->bf->reg;

    if (mqp->bf->uuarn > 0) {
        qp_info->dv.bf.size = mqp->bf->buf_size;
    } else {
        qp_info->dv.bf.size = 0; /* No BF */
    }
#endif
    return UCS_OK;
}

ucs_status_t uct_ib_mlx5_get_srq_info(struct ibv_srq *srq,
                                      uct_ib_mlx5dv_srq_t *srq_info)
{
#if HAVE_DECL_IBV_MLX5_EXP_GET_SRQ_INFO
    struct ibv_mlx5_srq_info ibv_srq_info;
    int ret;

    ret = ibv_mlx5_exp_get_srq_info(srq, &ibv_srq_info);
    if (ret != 0) {
        uct_ib_mlx5_obj_error("srq");
        return UCS_ERR_NO_DEVICE;
    }

    srq_info->dv.buf    = ibv_srq_info.buf;
    srq_info->dv.dbrec  = ibv_srq_info.dbrec;
    srq_info->dv.stride = ibv_srq_info.stride;
    srq_info->dv.head   = ibv_srq_info.head;
    srq_info->dv.tail   = ibv_srq_info.tail;
#else
    struct mlx5_srq *msrq;

    if (srq->handle == LEGACY_XRC_SRQ_HANDLE) {
        srq = (struct ibv_srq *)(((struct ibv_srq_legacy *)srq)->ibv_srq);
    }

    msrq = ucs_container_of(srq, struct mlx5_srq, vsrq.srq);

    if (msrq->counter != 0) {
        ucs_error("SRQ counter is not 0 (%d)", msrq->counter);
        return UCS_ERR_NO_DEVICE;
    }

    srq_info->dv.buf    = msrq->buf.buf;
    srq_info->dv.dbrec  = msrq->db;
    srq_info->dv.stride = 1 << msrq->wqe_shift;
    srq_info->dv.head   = msrq->head;
    srq_info->dv.tail   = msrq->tail;
#endif
    return UCS_OK;
}

static ucs_status_t uct_ib_mlx5_get_cq(struct ibv_cq *cq, uct_ib_mlx5dv_cq_t *mlx5_cq)
{
#if HAVE_DECL_IBV_MLX5_EXP_GET_CQ_INFO
    struct ibv_mlx5_cq_info ibv_cq_info;
    int ret;

    ret = ibv_mlx5_exp_get_cq_info(cq, &ibv_cq_info);
    if (ret != 0) {
        uct_ib_mlx5_obj_error("cq");
        return UCS_ERR_NO_DEVICE;
    }

    mlx5_cq->dv.buf      = ibv_cq_info.buf;
    mlx5_cq->dv.cqe_cnt  = ibv_cq_info.cqe_cnt;
    mlx5_cq->dv.cqn      = ibv_cq_info.cqn;
    mlx5_cq->dv.cqe_size = ibv_cq_info.cqe_size;
#else
    struct mlx5_cq *mcq = ucs_container_of(cq, struct mlx5_cq, ibv_cq);
    int ret;

    if (mcq->cons_index != 0) {
        ucs_error("CQ consumer index is not 0 (%d)", mcq->cons_index);
        return UCS_ERR_NO_DEVICE;
    }

    mlx5_cq->dv.buf      = mcq->active_buf->buf;
    mlx5_cq->dv.cqe_cnt  = mcq->ibv_cq.cqe + 1;
    mlx5_cq->dv.cqn      = mcq->cqn;
    mlx5_cq->dv.cqe_size = mcq->cqe_sz;
#endif
    return UCS_OK;
}

ucs_status_t uct_ib_mlx5dv_init_obj(uct_ib_mlx5dv_t *obj, uint64_t obj_type)
{
    ucs_status_t ret = UCS_OK;

    if (obj_type & MLX5DV_OBJ_QP) {
        ret = uct_ib_mlx5_get_qp_info(obj->dv.qp.in,
                ucs_container_of(obj->dv.qp.out, uct_ib_mlx5dv_qp_t, dv));
    }

    if (!ret && (obj_type & MLX5DV_OBJ_CQ)) {
        ret = uct_ib_mlx5_get_cq(obj->dv.cq.in,
                ucs_container_of(obj->dv.cq.out, uct_ib_mlx5dv_cq_t, dv));
    }

    if (!ret && (obj_type & MLX5DV_OBJ_SRQ)) {
        ret = uct_ib_mlx5_get_srq_info(obj->dv.srq.in,
                ucs_container_of(obj->dv.srq.out, uct_ib_mlx5dv_srq_t, dv));
    }

    return ret;
}
#endif

void uct_ib_mlx5_update_cq_ci(struct ibv_cq *cq, unsigned cq_ci)
{
#if HAVE_DECL_IBV_MLX5_EXP_UPDATE_CQ_CI
    ibv_mlx5_exp_update_cq_ci(cq, cq_ci);
#else
    struct mlx5_cq *mcq = ucs_container_of(cq, struct mlx5_cq, ibv_cq);
    mcq->cons_index = cq_ci;
#endif
}

unsigned uct_ib_mlx5_get_cq_ci(struct ibv_cq *cq)
{
    struct mlx5_cq *mcq = ucs_container_of(cq, struct mlx5_cq, ibv_cq);
    return mcq->cons_index;
}

#if !HAVE_DECL_MLX5DV_OBJ_AH
void uct_ib_mlx5_get_av(struct ibv_ah *ah, struct mlx5_wqe_av *av)
{
    memcpy(av, &ucs_container_of(ah, struct mlx5_ah, ibv_ah)->av, sizeof(*av));
}
#endif

struct ibv_qp *uct_dv_get_cmd_qp(struct ibv_srq *srq)
{
#if HAVE_STRUCT_MLX5_SRQ_CMD_QP
    struct mlx5_srq *msrq;

    if (srq->handle == LEGACY_XRC_SRQ_HANDLE) {
        srq = (struct ibv_srq *)(((struct ibv_srq_legacy *)srq)->ibv_srq);
    }

    msrq = ucs_container_of(srq, struct mlx5_srq, vsrq.srq);
    if (msrq->counter != 0) {
        ucs_error("SRQ counter is not 0 (%d)", msrq->counter);
        return NULL;
    }

    return &msrq->cmd_qp->verbs_qp.qp;
#else
    return NULL;
#endif
}

struct mlx5_uar_data {
    enum { __DUMMY }            map_type;
    void                        *regs;
};

void *uct_dv_get_info_uar0(void *uar)
{
#if HAVE_DECL_MLX5DV_INIT_OBJ
    struct mlx5_uar_data *muar = uar;
    return muar[0].regs;
#else
    return NULL;
#endif
}

#endif
