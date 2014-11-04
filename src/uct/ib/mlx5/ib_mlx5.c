/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ib_mlx5.h"

#include <ucs/sys/compiler.h>
#include <string.h>


ucs_status_t uct_ib_mlx5_get_qp_info(struct ibv_qp *qp, uct_ib_mlx5_qp_info_t *qp_info)
{
    struct mlx5_qp *mqp = ucs_container_of(qp, struct mlx5_qp, verbs_qp.qp);

    if ((mqp->sq.cur_post != 0) || (mqp->rq.head != 0) || mqp->bf->need_lock) {
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

    return UCS_OK;
}

ucs_status_t uct_ib_mlx5_get_cq_info(struct ibv_cq *cq, uct_ib_mlx5_cq_info_t *cq_info)
{
    struct mlx5_cq *mcq = ucs_container_of(cq, struct mlx5_cq, ibv_cq);

    if (mcq->cons_index != 0) {
        return UCS_ERR_NO_DEVICE;
    }

    cq_info->cqn      = mcq->cqn;
    cq_info->cqe_cnt  = mcq->ibv_cq.cqe + 1;
    cq_info->cqe_size = mcq->cqe_sz;
    cq_info->buf      = mcq->active_buf->buf;
    cq_info->dbrec    = mcq->dbrec;

    return UCS_OK;
}

void uct_ib_mlx5_update_cq_ci(struct ibv_cq *cq, unsigned cq_ci)
{
    struct mlx5_cq *mcq = ucs_container_of(cq, struct mlx5_cq, ibv_cq);

    mcq->cons_index = cq_ci;
}

void uct_ib_mlx5_get_av(struct ibv_ah *ah, struct mlx5_wqe_av *av)
{
    memcpy(av, &ucs_container_of(ah, struct mlx5_ah, ibv_ah)->av, sizeof(*av));
}

