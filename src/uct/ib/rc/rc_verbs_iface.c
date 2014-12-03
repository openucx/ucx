/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_verbs.h"

#include <uct/api/uct.h>
#include <uct/ib/base/ib_context.h>
#include <uct/tl/context.h>
#include <ucs/debug/log.h>
#include <string.h>

#define UCT_IB_MAX_WC 32


static UCS_F_NOINLINE void
uct_rc_verbs_iface_post_recv_always(uct_rc_verbs_iface_t *iface, unsigned count)
{
    struct ibv_recv_wr *wrs, *bad_wr;
    struct ibv_sge *sges;
    uct_ib_iface_recv_desc_t *desc;
    unsigned i;
    int ret;

    wrs  = alloca(sizeof *wrs  * count);
    sges = alloca(sizeof *sges * count);

    for (i = 0; i < count; ++i) {
        desc = ucs_mpool_get(iface->super.rx.mp);
        if (desc == NULL) {
            break;
        }

        sges[i].addr   = (uintptr_t)((void*)(desc) + iface->super.super.config.rx_hdr_offset);
        sges[i].length = iface->super.super.config.rx_data_size;
        sges[i].lkey   = desc->lkey;
        wrs[i].num_sge = 1;
        wrs[i].wr_id   = (uintptr_t)desc;
        wrs[i].sg_list = &sges[i];
        wrs[i].next    = &wrs[i + 1];
    }

    wrs[i - 1].next = NULL;

    ret = ibv_post_srq_recv(iface->super.rx.srq, wrs, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_post_srq_recv() returned %d: %m", ret);
    }

    iface->rx.available -= count;
}

static inline void uct_rc_verbs_iface_post_recv(uct_rc_verbs_iface_t *iface,
                                                int fill)
{
    unsigned batch = iface->super.config.rx_max_batch;
    unsigned count;

    if (iface->rx.available < batch) {
        if (!fill) {
            return;
        } else {
            count = iface->rx.available;
        }
    } else {
        count = batch;
    }

    uct_rc_verbs_iface_post_recv_always(iface, batch);
}

static void uct_rc_verbs_iface_poll_tx(uct_rc_verbs_iface_t *iface)
{
    uct_rc_verbs_ep_t *ep;
    struct ibv_wc wc[UCT_IB_MAX_WC];
    int i, ret;

    ret = ibv_poll_cq(iface->super.super.send_cq, UCT_IB_MAX_WC, wc);
    if (ret > 0) {
        for (i = 0; i < ret; ++i) {
            if (ucs_unlikely(wc[i].status != IBV_WC_SUCCESS)) {
                ucs_fatal("Send completion with error: %s", ibv_wc_status_str(wc[i].status));
            }

            ep = ucs_derived_of(uct_rc_iface_lookup_ep(&iface->super, wc[i].qp_num), uct_rc_verbs_ep_t);
            ucs_assert(ep != NULL);
            ep->tx.available += wc[i].wr_id;
        }
        iface->super.tx.outstanding -= ret;
    } else if (ucs_unlikely(ret < 0)) {
        ucs_fatal("Failed to poll send CQ");
    }
}

static void uct_rc_verbs_iface_poll_rx(uct_rc_verbs_iface_t *iface)
{
    uct_ib_iface_recv_desc_t *desc;
    uct_rc_hdr_t *hdr;
    ucs_status_t status;
    struct ibv_wc wc[UCT_IB_MAX_WC];
    int i, ret;

    ret = ibv_poll_cq(iface->super.super.recv_cq, UCT_IB_MAX_WC, wc);
    if (ret > 0) {
        for (i = 0; i < ret; ++i) {
            if (ucs_unlikely(wc[i].status != IBV_WC_SUCCESS)) {
                ucs_fatal("Receive completion with error: %s", ibv_wc_status_str(wc[i].status));
            }

            desc = (void*)wc[i].wr_id;
            hdr  = (void*)desc + iface->super.super.config.rx_hdr_offset;

            ucs_trace_data("RX: AM [%d]", hdr->am_id);

            status = uct_iface_invoke_am(&iface->super.super.super, hdr->am_id,
                                         hdr + 1, wc[i].byte_len - sizeof(*hdr));
            if (status == UCS_OK) {
                ucs_mpool_put(desc);
            }
        }

        iface->rx.available += ret;
    } else if (ret == 0) {
        uct_rc_verbs_iface_post_recv(iface, 0);
    } else if (ucs_unlikely(ret < 0)) {
        ucs_fatal("Failed to poll receive CQ");
    }
}

static void uct_rc_verbs_iface_progress(void *arg)
{
    uct_rc_verbs_iface_t *iface = arg;

    uct_rc_verbs_iface_poll_tx(iface);
    uct_rc_verbs_iface_poll_rx(iface);
}

static ucs_status_t uct_rc_verbs_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);

    uct_rc_iface_query(iface, iface_attr);
    iface_attr->max_short = 50;  /* TODO max_inline */
    return UCS_OK;
}

static UCS_CLASS_INIT_FUNC(uct_rc_verbs_iface_t, uct_context_h context,
                           const char *dev_name, size_t rx_headroom,
                           uct_iface_config_t *tl_config)
{
    uct_rc_iface_config_t *config = ucs_derived_of(tl_config, uct_rc_iface_config_t);

    extern uct_iface_ops_t uct_rc_verbs_iface_ops;
    UCS_CLASS_CALL_SUPER_INIT(&uct_rc_verbs_iface_ops, context, dev_name,
                              rx_headroom, tl_config);

    /* Initialize inline work request */
    self->inl_am_wr.wr_id                   = 0;
    self->inl_am_wr.next                    = NULL;
    self->inl_am_wr.sg_list                 = self->inl_sge;
    self->inl_am_wr.num_sge                 = 2;
    self->inl_am_wr.opcode                  = IBV_WR_SEND;
    self->inl_am_wr.send_flags              = IBV_SEND_INLINE;
    self->inl_am_wr.imm_data                = 0;
    self->inl_rwrite_wr.wr_id               = 0;
    self->inl_rwrite_wr.next                = NULL;
    self->inl_rwrite_wr.sg_list             = self->inl_sge;
    self->inl_rwrite_wr.num_sge             = 1;
    self->inl_rwrite_wr.opcode              = IBV_WR_RDMA_WRITE;
    self->inl_rwrite_wr.send_flags          = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    self->inl_rwrite_wr.imm_data            = 0;
    self->inl_rwrite_wr.wr.rdma.remote_addr = 0;
    self->inl_rwrite_wr.wr.rdma.rkey        = 0;
    self->inl_sge[0].addr                   = 0;
    self->inl_sge[0].length                 = 0;
    self->inl_sge[0].lkey                   = 0;
    self->inl_sge[1].addr                   = 0;
    self->inl_sge[1].length                 = 0;
    self->inl_sge[1].lkey                   = 0;
    self->rx.available                      = config->super.rx.queue_len;

    while (self->rx.available > 0) {
        uct_rc_verbs_iface_post_recv(self, 1);
    }

    ucs_notifier_chain_add(&context->progress_chain, uct_rc_verbs_iface_progress,
                           self);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_verbs_iface_t)
{
    uct_context_h context = self->super.super.super.pd->context;

    ucs_notifier_chain_remove(&context->progress_chain, uct_rc_verbs_iface_progress, self);
}

UCS_CLASS_DEFINE(uct_rc_verbs_iface_t, uct_rc_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_verbs_iface_t, uct_iface_t, uct_context_h,
                                 const char*, size_t, uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_verbs_iface_t, uct_iface_t);


uct_iface_ops_t uct_rc_verbs_iface_ops = {
    .iface_close         = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_verbs_iface_t),
    .iface_get_address   = uct_rc_iface_get_address,
    .iface_flush         = uct_rc_iface_flush,
    .ep_get_address      = uct_rc_ep_get_address,
    .ep_connect_to_iface = NULL,
    .ep_connect_to_ep    = uct_rc_ep_connect_to_ep,
    .iface_query         = uct_rc_verbs_iface_query,
    .ep_am_short         = uct_rc_verbs_ep_am_short,
    .ep_put_short        = uct_rc_verbs_ep_put_short,
    .ep_create           = UCS_CLASS_NEW_FUNC_NAME(uct_rc_verbs_ep_t),
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_verbs_ep_t),
};


static ucs_status_t uct_rc_verbs_query_resources(uct_context_h context,
                                                 uct_resource_desc_t **resources_p,
                                                 unsigned *num_resources_p)
{
    /* TODO take transport overhead into account */
    return uct_ib_query_resources(context, 0, resources_p, num_resources_p);
}

static uct_tl_ops_t uct_rc_verbs_tl_ops = {
    .query_resources     = uct_rc_verbs_query_resources,
    .iface_open          = UCS_CLASS_NEW_FUNC_NAME(uct_rc_verbs_iface_t),
    .rkey_unpack         = uct_ib_rkey_unpack,
};

static void uct_rc_verbs_register(uct_context_t *context)
{
    uct_register_tl(context, "rc_verbs", uct_rc_iface_config_table,
                    sizeof(uct_rc_iface_config_t), &uct_rc_verbs_tl_ops);
}

UCS_COMPONENT_DEFINE(uct_context_t, rc_verbs, uct_rc_verbs_register, ucs_empty_function, 0)
