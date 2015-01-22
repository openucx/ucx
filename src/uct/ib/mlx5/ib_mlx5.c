/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ib_mlx5.h"

#include <uct/ib/base/ib_verbs.h>
#include <ucs/debug/log.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/sys.h>
#include <string.h>


ucs_status_t uct_ib_mlx5_get_qp_info(struct ibv_qp *qp, uct_ib_mlx5_qp_info_t *qp_info)
{
#if HAVE_DECL_IBV_MLX5_EXP_GET_QP_INFO
    struct ibv_mlx5_qp_info ibv_qp_info;
    int ret;

    ret = ibv_mlx5_exp_get_qp_info(qp, &ibv_qp_info);
    if (ret != 0) {
        return UCS_ERR_NO_DEVICE;
    }

    qp_info->qpn        = ibv_qp_info.qpn;
    qp_info->dbrec      = ibv_qp_info.dbrec;
    qp_info->sq.buf     = ibv_qp_info.sq.buf;
    qp_info->sq.wqe_cnt = ibv_qp_info.sq.wqe_cnt;
    qp_info->sq.stride  = ibv_qp_info.sq.stride;
    qp_info->rq.buf     = ibv_qp_info.rq.buf;
    qp_info->rq.wqe_cnt = ibv_qp_info.rq.wqe_cnt;
    qp_info->rq.stride  = ibv_qp_info.rq.stride;
    qp_info->bf.reg     = ibv_qp_info.bf.reg;
    qp_info->bf.size    = ibv_qp_info.bf.size;
#else
    struct mlx5_qp *mqp = ucs_container_of(qp, struct mlx5_qp, verbs_qp.qp);

    if ((mqp->sq.cur_post != 0) || (mqp->rq.head != 0) || mqp->bf->need_lock) {
        ucs_warn("cur_post=%d head=%d need_lock=%d", mqp->sq.cur_post,
                 mqp->rq.head, mqp->bf->need_lock);
        return UCS_ERR_NO_DEVICE;
    }

    qp_info->qpn        = qp->qp_num;
    qp_info->dbrec      = mqp->db;
    qp_info->sq.buf     = mqp->buf.buf + mqp->sq.offset;
    qp_info->sq.wqe_cnt = mqp->sq.wqe_cnt;
    qp_info->sq.stride  = 1 << mqp->sq.wqe_shift;
    qp_info->rq.buf     = mqp->buf.buf + mqp->rq.offset;
    qp_info->rq.wqe_cnt = mqp->rq.wqe_cnt;
    qp_info->rq.stride  = 1 << mqp->rq.wqe_shift;
    qp_info->bf.reg     = mqp->bf->reg;

    if (mqp->bf->uuarn > 0) {
        qp_info->bf.size = mqp->bf->buf_size;
    } else {
        qp_info->bf.size = 0; /* No BF */
    }
#endif
    return UCS_OK;
}

ucs_status_t uct_ib_mlx5_get_srq_info(struct ibv_srq *srq, uct_ib_mlx5_srq_info_t *srq_info)
{
#if HAVE_DECL_IBV_MLX5_EXP_GET_SRQ_INFO
    struct ibv_mlx5_srq_info ibv_srq_info;
    int ret;

    ret = ibv_mlx5_exp_get_srq_info(srq, &ibv_srq_info);
    if (ret != 0) {
        return UCS_ERR_NO_DEVICE;
    }

    srq_info->buf    = ibv_srq_info.buf;
    srq_info->dbrec  = ibv_srq_info.dbrec;
    srq_info->stride = ibv_srq_info.stride;
    srq_info->head   = ibv_srq_info.head;
    srq_info->tail   = ibv_srq_info.tail;
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

    srq_info->buf    = msrq->buf.buf;
    srq_info->dbrec  = msrq->db;
    srq_info->stride = 1 << msrq->wqe_shift;
    srq_info->head   = msrq->head;
    srq_info->tail   = msrq->tail;
#endif
    return UCS_OK;
}

ucs_status_t uct_ib_mlx5_get_cq(struct ibv_cq *cq, uct_ib_mlx5_cq_t *mlx5_cq)
{
#if HAVE_DECL_IBV_MLX5_EXP_GET_CQ_INFO
    struct ibv_mlx5_cq_info ibv_cq_info;
    int ret;

    ret = ibv_mlx5_exp_get_cq_info(cq, &ibv_cq_info);
    if (ret != 0) {
        return UCS_ERR_NO_DEVICE;
    }

    mlx5_cq->cq_buf    = ibv_cq_info.buf;
    mlx5_cq->cq_ci     = 0;
    mlx5_cq->cq_length = ibv_cq_info.cqe_cnt;
    mlx5_cq->cqe_size  = ibv_cq_info.cqe_size;
#else
    struct mlx5_cq *mcq = ucs_container_of(cq, struct mlx5_cq, ibv_cq);
    int ret;

    if (mcq->cons_index != 0) {
        ucs_error("CQ consumer index is not 0 (%d)", mcq->cons_index);
        return UCS_ERR_NO_DEVICE;
    }

    ret = ibv_exp_cq_ignore_overrun(cq);
    if (ret != 0) {
        ucs_error("Failed to modify send CQ to ignore overrun: %s", strerror(ret));
        return UCS_ERR_UNSUPPORTED;
    }

    mlx5_cq->cq_buf      = mcq->active_buf->buf;
    mlx5_cq->cq_ci       = 0;
    mlx5_cq->cq_length   = mcq->ibv_cq.cqe + 1;
    mlx5_cq->cqe_size    = mcq->cqe_sz;
#endif
    return UCS_OK;
}

void uct_ib_mlx5_update_cq_ci(struct ibv_cq *cq, unsigned cq_ci)
{
#if HAVE_DECL_IBV_MLX5_EXP_UPDATE_CQ_CI
    ibv_mlx5_exp_update_cq_ci(cq, cq_ci);
#else
    struct mlx5_cq *mcq = ucs_container_of(cq, struct mlx5_cq, ibv_cq);
    mcq->cons_index = cq_ci;
#endif
}

void uct_ib_mlx5_get_av(struct ibv_ah *ah, struct mlx5_wqe_av *av)
{
    memcpy(av, &ucs_container_of(ah, struct mlx5_ah, ibv_ah)->av, sizeof(*av));
}

void uct_ib_mlx5_check_completion(struct mlx5_cqe64 *cqe)
{
    struct mlx5_err_cqe *ecqe;

    switch (cqe->op_own >> 4) {
    case MLX5_CQE_INVALID:
        break;
    case MLX5_CQE_REQ_ERR:
    case MLX5_CQE_RESP_ERR:
        ecqe = (void*)cqe;
        ucs_fatal("completion with error synd %d vend %d wcnt %d",
                  ecqe->syndrome, ecqe->vendor_err_synd, ntohs(ecqe->wqe_counter));
        break;
    default:
        break;
    }
}
