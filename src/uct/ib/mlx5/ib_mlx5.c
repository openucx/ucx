/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ib_mlx5.h"
#include "ib_mlx5.inl"
#include "ib_mlx5_log.h"

#include <uct/ib/base/ib_verbs.h>
#include <uct/ib/base/ib_device.h>
#include <ucs/arch/bitops.h>
#include <ucs/debug/log.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/sys.h>
#include <string.h>

ucs_status_t uct_ib_mlx5_get_compact_av(uct_ib_iface_t *iface, int *compact_av)
{
    struct mlx5_wqe_av  mlx5_av;
    struct ibv_ah      *ah;
    uct_ib_address_t   *ib_addr;
    ucs_status_t        status;
    struct ibv_ah_attr  ah_attr;

    /* coverity[result_independent_of_operands] */
    ib_addr = ucs_alloca((size_t)iface->addr_size);

    status = uct_ib_iface_get_device_address(&iface->super.super,
                                             (uct_device_addr_t*)ib_addr);
    if (status != UCS_OK) {
        return status;
    }

    uct_ib_iface_fill_ah_attr_from_addr(iface, ib_addr, iface->path_bits[0], &ah_attr);
    status = uct_ib_iface_create_ah(iface, &ah_attr, &ah);
    if (status != UCS_OK) {
        return status;
    }

    uct_ib_mlx5_get_av(ah, &mlx5_av);
    ibv_destroy_ah(ah);

    /* copy MLX5_EXTENDED_UD_AV from the driver, if the flag is not present then
     * the device supports compact address vector. */
    *compact_av = !(mlx5_av_base(&mlx5_av)->dqp_dct & UCT_IB_MLX5_EXTENDED_UD_AV);
    return UCS_OK;
}


void uct_ib_mlx5_check_completion(uct_ib_iface_t *iface, uct_ib_mlx5_cq_t *cq,
                                  struct mlx5_cqe64 *cqe)
{
    ucs_status_t status;

    switch (cqe->op_own >> 4) {
    case MLX5_CQE_REQ_ERR:
        /* update ci before invoking error callback, since it can poll on cq */
        UCS_STATIC_ASSERT(MLX5_CQE_REQ_ERR & (UCT_IB_MLX5_CQE_OP_OWN_ERR_MASK >> 4));
        ++cq->cq_ci;
        status = uct_ib_mlx5_completion_with_err((void*)cqe, UCS_LOG_LEVEL_DEBUG);
        iface->ops->handle_failure(iface, cqe, status);
        return;
    case MLX5_CQE_RESP_ERR:
        /* Local side failure - treat as fatal */
        UCS_STATIC_ASSERT(MLX5_CQE_RESP_ERR & (UCT_IB_MLX5_CQE_OP_OWN_ERR_MASK >> 4));
        ++cq->cq_ci;
        uct_ib_mlx5_completion_with_err((void*)cqe, UCS_LOG_LEVEL_FATAL);
        return;
    default:
        /* CQE might have been updated by HW. Skip it now, and it would be handled
         * in next polling. */
        return;
    }
}

static int uct_ib_mlx5_bf_cmp(uct_ib_mlx5_bf_t *bf, uintptr_t addr, unsigned bf_size)
{
    return (bf->reg.addr & ~UCT_IB_MLX5_BF_REG_SIZE) == (addr & ~UCT_IB_MLX5_BF_REG_SIZE);
}

static ucs_status_t uct_ib_mlx5_bf_init(uct_ib_mlx5_bf_t *bf, uintptr_t addr,
                                        unsigned bf_size)
{
    bf->reg.addr  = addr;
    bf->enable_bf = bf_size;
    return UCS_OK;
}

static void uct_ib_mlx5_bf_cleanup(uct_ib_mlx5_bf_t *bf)
{
}

void uct_ib_mlx5_txwq_reset(uct_ib_mlx5_txwq_t *txwq)
{
    txwq->curr       = txwq->qstart;
    txwq->sw_pi      = 0;
    txwq->prev_sw_pi = -1;
#if ENABLE_ASSERT
    txwq->hw_ci      = 0xFFFF;
#endif
    memset(txwq->qstart, 0, txwq->qend - txwq->qstart);
}

ucs_status_t uct_ib_mlx5_txwq_init(uct_priv_worker_t *worker,
                                   uct_ib_mlx5_txwq_t *txwq,
                                   struct ibv_qp *verbs_qp)
{
    uct_ib_mlx5_qp_info_t qp_info;
    ucs_status_t status;

    status = uct_ib_mlx5_get_qp_info(verbs_qp, &qp_info);
    if (status != UCS_OK) {
        return UCS_ERR_IO_ERROR;
    }

    if ((qp_info.sq.stride != MLX5_SEND_WQE_BB) || !ucs_is_pow2(qp_info.sq.wqe_cnt) ||
        ((qp_info.bf.size != 0) && (qp_info.bf.size != UCT_IB_MLX5_BF_REG_SIZE)))
    {
        ucs_error("mlx5 device parameters not suitable for transport "
                  "bf.size(%d) %d, sq.stride(%d) %d, wqe_cnt %d",
                  UCT_IB_MLX5_BF_REG_SIZE, qp_info.bf.size,
                  MLX5_SEND_WQE_BB, qp_info.sq.stride, qp_info.sq.wqe_cnt);
        return UCS_ERR_IO_ERROR;
    }

    ucs_debug("tx wq %d bytes [bb=%d, nwqe=%d]",
              qp_info.sq.stride * qp_info.sq.wqe_cnt,
              qp_info.sq.stride, qp_info.sq.wqe_cnt);

    txwq->qstart   = qp_info.sq.buf;
    txwq->qend     = qp_info.sq.buf + (qp_info.sq.stride * qp_info.sq.wqe_cnt);
    txwq->bf       = uct_worker_tl_data_get(worker,
                                            UCT_IB_MLX5_WORKER_BF_KEY,
                                            uct_ib_mlx5_bf_t,
                                            uct_ib_mlx5_bf_cmp,
                                            uct_ib_mlx5_bf_init,
                                            (uintptr_t)qp_info.bf.reg,
                                            qp_info.bf.size);
    if (UCS_PTR_IS_ERR(txwq->bf)) {
        return UCS_PTR_STATUS(txwq->bf);
    }

    txwq->dbrec    = &qp_info.dbrec[MLX5_SND_DBR];
    /* need to reserve 2x because:
     *  - on completion we only get the index of last wqe and we do not
     *    really know how many bb is there (but no more than max bb
     *  - on send we check that there is at least one bb. We know
     *  exact number of bbs once we actually are sending.
     */
    txwq->bb_max   = qp_info.sq.wqe_cnt - 2 * UCT_IB_MLX5_MAX_BB;
    ucs_assert_always(txwq->bb_max > 0);

    uct_ib_mlx5_txwq_reset(txwq);
    return UCS_OK;
}

void uct_ib_mlx5_txwq_cleanup(uct_ib_mlx5_txwq_t* txwq)
{
    uct_worker_tl_data_put(txwq->bf, uct_ib_mlx5_bf_cleanup);
}

ucs_status_t uct_ib_mlx5_get_rxwq(struct ibv_qp *verbs_qp, uct_ib_mlx5_rxwq_t *rxwq)
{
    uct_ib_mlx5_qp_info_t qp_info;
    ucs_status_t status;

    status = uct_ib_mlx5_get_qp_info(verbs_qp, &qp_info);
    if (status != UCS_OK) {
        return UCS_ERR_IO_ERROR;
    }

    if (!ucs_is_pow2(qp_info.rq.wqe_cnt) ||
        qp_info.rq.stride != sizeof(struct mlx5_wqe_data_seg)) {
        ucs_error("mlx5 rx wq [count=%d stride=%d] has invalid parameters",
                  qp_info.rq.wqe_cnt,
                  qp_info.rq.stride);
        return UCS_ERR_IO_ERROR;
    }
    rxwq->wqes            = qp_info.rq.buf;
    rxwq->rq_wqe_counter  = 0;
    rxwq->cq_wqe_counter  = 0;
    rxwq->mask            = qp_info.rq.wqe_cnt - 1;
    rxwq->dbrec           = &qp_info.dbrec[MLX5_RCV_DBR];
    memset(rxwq->wqes, 0, qp_info.rq.wqe_cnt * sizeof(struct mlx5_wqe_data_seg));

    return UCS_OK;
}

ucs_status_t uct_ib_mlx5_srq_init(uct_ib_mlx5_srq_t *srq, struct ibv_srq *verbs_srq,
                                  size_t sg_byte_count)
{
    uct_ib_mlx5_srq_info_t srq_info;
    uct_ib_mlx5_srq_seg_t *seg;
    ucs_status_t status;
    unsigned i;

    status = uct_ib_mlx5_get_srq_info(verbs_srq, &srq_info);
    if (status != UCS_OK) {
        return status;
    }

    if (srq_info.head != 0) {
        ucs_error("SRQ head is not 0 (%d)", srq_info.head);
        return UCS_ERR_NO_DEVICE;
    }

    if (srq_info.stride != UCT_IB_MLX5_SRQ_STRIDE) {
        ucs_error("SRQ stride is not %lu (%d)", UCT_IB_MLX5_SRQ_STRIDE,
                  srq_info.stride);
        return UCS_ERR_NO_DEVICE;
    }

    if (!ucs_is_pow2(srq_info.tail + 1)) {
        ucs_error("SRQ length is not power of 2 (%d)", srq_info.tail + 1);
        return UCS_ERR_NO_DEVICE;
    }

    srq->buf             = srq_info.buf;
    srq->db              = srq_info.dbrec;
    srq->free_idx        = srq_info.tail;
    srq->ready_idx       = -1;
    srq->sw_pi           = -1;
    srq->mask            = srq_info.tail;
    srq->tail            = srq_info.tail;

    for (i = srq_info.head; i <= srq_info.tail; ++i) {
        seg = uct_ib_mlx5_srq_get_wqe(srq, i);
        seg->srq.free        = 0;
        seg->srq.desc        = NULL;
        seg->dptr.byte_count = htonl(sg_byte_count);
    }

    return UCS_OK;
}

void uct_ib_mlx5_srq_cleanup(uct_ib_mlx5_srq_t *srq, struct ibv_srq *verbs_srq)
{
    uct_ib_mlx5_srq_info_t srq_info;
    ucs_status_t status;

    status = uct_ib_mlx5_get_srq_info(verbs_srq, &srq_info);
    ucs_assert_always(status == UCS_OK);
    ucs_assertv_always(srq->tail == srq_info.tail, "srq->tail=%d srq_info.tail=%d",
                       srq->tail, srq_info.tail);
}
