/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_iface.h"
#include "rc_mlx5.h"

#include <uct/api/uct.h>
#include <uct/ib/base/ib_context.h>
#include <uct/ib/base/ib_iface.h>
#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/tl/context.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>
#include <string.h>
#include <arpa/inet.h> /* For htonl */


typedef struct {
    struct mlx5_wqe_ctrl_seg     ctrl;
    struct mlx5_wqe_raddr_seg    raddr;
    struct mlx5_wqe_inl_data_seg inl;
} UCS_S_PACKED uct_ib_mlx5_wqe_rc_rdma_inl_seg_t;


static ucs_status_t uct_rc_mlx5_query_resources(uct_context_h context,
                                                uct_resource_desc_t **resources_p,
                                                unsigned *num_resources_p)
{
    /* TODO take transport overhead into account */
    return uct_ib_query_resources(context, UCT_IB_RESOURCE_FLAG_MLX5_PRM,
                                  resources_p, num_resources_p);
}

static UCS_CLASS_INIT_FUNC(uct_rc_mlx5_ep_t, uct_iface_h tl_iface)
{
    uct_ib_mlx5_qp_info_t qp_info;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(tl_iface);

    status = uct_ib_mlx5_get_qp_info(self->super.qp, &qp_info);
    if (status != UCS_OK) {
        ucs_error("Failed to get mlx5 QP information");
        return status;
    }

    if ((qp_info.bf.size == 0) || !ucs_is_pow2(qp_info.bf.size) ||
        (qp_info.sq.stride != MLX5_SEND_WQE_BB) || !ucs_is_pow2(qp_info.sq.wqe_cnt))
    {
        ucs_error("mlx5 device parameters not suitable for transport");
        return status;
    }

    self->qpn_ds     = htonl(self->super.qp->qp_num << 8);
    self->tx.qstart  = qp_info.sq.buf;
    self->tx.qend    = qp_info.sq.buf + (MLX5_SEND_WQE_BB *  qp_info.sq.wqe_cnt);
    self->tx.seg     = self->tx.qstart;
    self->tx.sw_pi   = 0;
    self->tx.max_pi  = UCT_RC_TX_QP_LEN;
    self->tx.bf_reg  = qp_info.bf.reg;
    self->tx.bf_size = qp_info.bf.size;
    self->tx.dbrec   = &qp_info.dbrec[MLX5_SND_DBR];

    memset(self->tx.qstart, 0, self->tx.qend - self->tx.qstart);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_mlx5_ep_t)
{
}

UCS_CLASS_DEFINE(uct_rc_mlx5_ep_t, uct_rc_ep_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_mlx5_ep_t, uct_ep_t, uct_iface_h);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_mlx5_ep_t, uct_ep_t);


static ucs_status_t uct_rc_mlx5_ep_put_short(uct_ep_h tl_ep, void *buffer,
                                             unsigned length,
                                             uint64_t remote_addr,
                                             uct_rkey_t rkey)
{
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    uct_ib_mlx5_wqe_rc_rdma_inl_seg_t *seg = ep->tx.seg;
    unsigned wqe_size;
    uint32_t sw_pi_16_n;
    uint64_t *src, *dst, *end;
    unsigned sw_pi, i;

    sw_pi = ep->tx.sw_pi;
    if (UCS_CIRCULAR_COMPARE32(sw_pi, >=, ep->tx.max_pi)) {
        return UCS_ERR_WOULD_BLOCK;
    }

    sw_pi_16_n = htonl(sw_pi & 0xffff);
    wqe_size   = (length + sizeof(*seg) + 15) / 16;
    ucs_assert(wqe_size <= MLX5_SEND_WQE_BB);

    /* Build WQE */
    seg->ctrl.opmod_idx_opcode   = (sw_pi_16_n >> 8) | (MLX5_OPCODE_RDMA_WRITE << 24);
    seg->ctrl.qpn_ds             = htonl(wqe_size) | ep->qpn_ds;
    seg->ctrl.fm_ce_se           = MLX5_WQE_CTRL_CQ_UPDATE; /* Ask for completion */
    seg->raddr.raddr             = htonll(remote_addr);
    seg->raddr.rkey              = (uint32_t)rkey;

    /* Data */
    UCS_STATIC_ASSERT(seg + 1 == ((void*)seg + 32 + 4));
    seg->inl.byte_count          = htonl(length | MLX5_INLINE_SEG);
    memcpy(seg + 1, buffer, length);

    /* Write doorbell record */
    ucs_compiler_fence();
    *ep->tx.dbrec = sw_pi_16_n;

    /* Make sure that doorbell record is written before ringing the doorbell */
    ucs_memory_bus_store_fence();

    /* Set up copy pointers */
    dst = ep->tx.bf_reg;
    src = (void*)seg;
    end = (void*)seg + sizeof(*seg) + length;

    /* BF copy */
    do {
        for (i = 0; i < MLX5_SEND_WQE_BB / sizeof(*dst); ++i) {
            *dst++ = *src++;
        }
    } while (src < end);

    /* Flip BF register */
    ep->tx.bf_reg = (void*) ((uintptr_t) ep->tx.bf_reg ^ ep->tx.bf_size);

    /* Completion counters */
    ++ep->tx.sw_pi;
    ++ucs_derived_of(ep->super.super.iface, uct_rc_mlx5_iface_t)->tx.outstanding;

    /* Advance queue pointer */
    if (ucs_unlikely((ep->tx.seg += MLX5_SEND_WQE_BB) >= ep->tx.qend)) {
        ep->tx.seg = ep->tx.qstart;
    }

    return UCS_OK;
}

static void uct_rc_mlx5_iface_progress(void *arg)
{
    uct_rc_mlx5_iface_t *iface = arg;
    struct mlx5_cqe64 *cqe;
    uct_rc_mlx5_ep_t *ep;
    unsigned index, qp_num;

    index = iface->tx.cq_ci;
    cqe   = iface->tx.cq_buf + (index & (iface->tx.cq_length - 1)) * sizeof(struct mlx5_cqe64);
    if (uct_ib_mlx5_cqe_hw_owned(cqe, index, iface->tx.cq_length)) {
        return; /* CQ is empty */
    }
    iface->tx.cq_ci = index + 1;
    --iface->tx.outstanding;

    ucs_memory_cpu_load_fence();

    qp_num = ntohl(cqe->sop_drop_qpn) & 0xffffff;
    ep = ucs_derived_of(uct_rc_iface_lookup_ep(&iface->super, qp_num), uct_rc_mlx5_ep_t);
    ucs_assert(ep != NULL);

    ++ep->tx.max_pi;
}

static ucs_status_t uct_rc_mlx5_iface_flush(uct_iface_h tl_iface, uct_req_h *req_p,
                                            uct_completion_cb_t cb)
{
    uct_rc_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_mlx5_iface_t);

    if (iface->tx.outstanding > 0) {
        return UCS_ERR_WOULD_BLOCK;
    }

    return UCS_OK;
}

static ucs_status_t uct_rc_mlx5_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);

    uct_rc_iface_query(iface, iface_attr);
    iface_attr->max_short = MLX5_SEND_WQE_BB - sizeof(uct_ib_mlx5_wqe_rc_rdma_inl_seg_t);  /* TODO */
    return UCS_OK;
}

static void UCS_CLASS_DELETE_FUNC_NAME(uct_rc_mlx5_iface_t)(uct_iface_t*);

uct_iface_ops_t uct_rc_mlx5_iface_ops = {
    .iface_close         = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_mlx5_iface_t),
    .iface_get_address   = uct_rc_iface_get_address,
    .iface_flush         = uct_rc_mlx5_iface_flush,
    .ep_get_address      = uct_rc_ep_get_address,
    .ep_connect_to_iface = NULL,
    .ep_connect_to_ep    = uct_rc_ep_connect_to_ep,
    .iface_query         = uct_rc_mlx5_iface_query,
    .ep_put_short        = uct_rc_mlx5_ep_put_short,
    .ep_create           = UCS_CLASS_NEW_FUNC_NAME(uct_rc_mlx5_ep_t),
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_mlx5_ep_t),
};

static UCS_CLASS_INIT_FUNC(uct_rc_mlx5_iface_t, uct_context_h context, const char *dev_name)
{
    uct_ib_mlx5_cq_info_t cq_info;
    ucs_status_t status;
    int ret;

    UCS_CLASS_CALL_SUPER_INIT(&uct_rc_mlx5_iface_ops, context, dev_name);

    ret = ibv_exp_cq_ignore_overrun(self->super.super.send_cq);
    if (ret != 0) {
        ucs_error("Failed to modify send CQ to ignore overrun: %s", strerror(ret));
        return UCS_ERR_UNSUPPORTED;
    }

    status = uct_ib_mlx5_get_cq_info(self->super.super.send_cq, &cq_info);
    if (status != UCS_OK) {
        ucs_error("Failed to get mlx5 CQ information");
        return status;
    }

    self->tx.cq_buf      = cq_info.buf;
    self->tx.cq_ci       = 0;
    self->tx.cq_length   = cq_info.cqe_cnt;
    self->tx.outstanding = 0;

    ucs_notifier_chain_add(&context->progress_chain, uct_rc_mlx5_iface_progress,
                           self);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_mlx5_iface_t)
{
    uct_context_h context = self->super.super.super.pd->context;
    ucs_notifier_chain_remove(&context->progress_chain, uct_rc_mlx5_iface_progress, self);
}

UCS_CLASS_DEFINE(uct_rc_mlx5_iface_t, uct_ib_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_mlx5_iface_t, uct_iface_t, uct_context_h, const char*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_mlx5_iface_t, uct_iface_t);


uct_tl_ops_t uct_rc_mlx5_tl_ops = {
    .query_resources     = uct_rc_mlx5_query_resources,
    .iface_open          = UCS_CLASS_NEW_FUNC_NAME(uct_rc_mlx5_iface_t),
    .rkey_unpack         = uct_ib_rkey_unpack,
};

