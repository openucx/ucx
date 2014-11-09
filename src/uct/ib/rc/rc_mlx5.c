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
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
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

static ucs_status_t uct_rc_mlx5_ep_create(uct_iface_h tl_iface, uct_ep_h *ep_p)
{
    uct_ib_mlx5_qp_info_t qp_info;
    uct_rc_mlx5_ep_t *ep;
    ucs_status_t status;

    ep = ucs_malloc(sizeof(*ep), "rc mlx5 ep");
    if (ep == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    ep->super.super.iface = tl_iface;

    status = uct_rc_ep_init(&ep->super);
    if (status != UCS_OK) {
        goto err_free;
    }

    status = uct_ib_mlx5_get_qp_info(ep->super.qp, &qp_info);
    if (status != UCS_OK) {
        goto err_cleanup_rc_ep;
    }

    if ((qp_info.bf.size == 0) || !ucs_is_pow2(qp_info.bf.size) ||
        (qp_info.sq.stride != MLX5_SEND_WQE_BB) || !ucs_is_pow2(qp_info.sq.wqe_cnt))
    {
        ucs_error("mlx5 device parameters not suitable for transport");
        goto err_cleanup_rc_ep;
    }

    ep->qpn_ds     = htonl(ep->super.qp->qp_num << 8);
    ep->tx.qstart  = qp_info.sq.buf;
    ep->tx.qend    = qp_info.sq.buf + (MLX5_SEND_WQE_BB *  qp_info.sq.wqe_cnt);
    ep->tx.seg     = ep->tx.qstart;
    ep->tx.sw_pi   = 0;
    ep->tx.hw_ci   = 0;
    ep->tx.bf_reg  = qp_info.bf.reg;
    ep->tx.bf_size = qp_info.bf.size;
    ep->tx.dbrec   = &qp_info.dbrec[MLX5_SND_DBR];

    memset(ep->tx.qstart, 0, ep->tx.qend - ep->tx.qstart);

    *ep_p = &ep->super.super;
    return UCS_OK;

err_cleanup_rc_ep:
    uct_rc_ep_cleanup(&ep->super);
err_free:
    ucs_free(ep);
err:
    return status;
}

static void uct_rc_mlx5_ep_destroy(uct_ep_h tl_ep)
{
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);

    uct_rc_ep_cleanup(&ep->super);
    ucs_free(ep);
}

static ucs_status_t uct_rc_mlx5_ep_put_short(uct_ep_h tl_ep, void *buffer,
                                             unsigned length,
                                             uint64_t remote_addr,
                                             uct_rkey_t rkey, uct_req_h *req_p,
                                             uct_completion_cb_t cb)
{
    uct_rc_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_mlx5_ep_t);
    uct_ib_mlx5_wqe_rc_rdma_inl_seg_t *seg = ep->tx.seg;
    unsigned wqe_size;
    uint32_t sw_pi_16_n;
    uint64_t *src, *dst;
    unsigned i;

    sw_pi_16_n = htonl(ep->tx.sw_pi & 0xffff);
    wqe_size = ((length + 15) / 16) + (sizeof(*seg) / 16);
    ucs_assert(wqe_size < MLX5_SEND_WQE_BB);

    /* Build WQE */
    seg->ctrl.opmod_idx_opcode   = (sw_pi_16_n >> 8) | (MLX5_OPCODE_RDMA_WRITE << 24);
    seg->ctrl.qpn_ds             = htonl(wqe_size) | ep->qpn_ds;
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

    /* BF copy */
    dst = ep->tx.bf_reg;
    src = (void*)seg;
    for (i = 0; i < MLX5_SEND_WQE_BB / sizeof(*dst); ++i) {
        *dst++ = *src++;
    }

    ucs_memory_bus_store_fence();

    /* Flip BF register */
    ep->tx.bf_reg = (void*) ((uintptr_t) ep->tx.bf_reg ^ ep->tx.bf_size);
    ++ep->tx.sw_pi;

    /* Advance queue pointer */
    ep->tx.seg += MLX5_SEND_WQE_BB;
    if (ep->tx.seg == ep->tx.qend) {
        ep->tx.seg = ep->tx.qstart;
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

static ucs_status_t uct_rc_mlx5_iface_open(uct_context_h context,
                                           const char *hw_name,
                                           uct_iface_h *iface_p)
{
    ucs_status_t status;
    uct_iface_h iface;

    status = uct_rc_iface_open(context, hw_name, &iface);
    if (status != UCS_OK) {
        return status;
    }

    iface->ops.iface_query         = uct_rc_mlx5_iface_query;
    iface->ops.ep_put_short        = uct_rc_mlx5_ep_put_short;
    iface->ops.ep_create           = uct_rc_mlx5_ep_create;
    iface->ops.ep_destroy          = uct_rc_mlx5_ep_destroy;
    *iface_p = iface;
    return UCS_OK;
}

uct_tl_ops_t uct_rc_mlx5_tl_ops = {
    .query_resources     = uct_rc_mlx5_query_resources,
    .iface_open          = uct_rc_mlx5_iface_open,
    .rkey_unpack         = uct_ib_rkey_unpack,
};

