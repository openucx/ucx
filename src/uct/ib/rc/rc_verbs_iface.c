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


ucs_config_field_t uct_rc_verbs_iface_config_table[] = {
  {"RC_", "", NULL,
   ucs_offsetof(uct_rc_verbs_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_rc_iface_config_table)},

  {"MAX_AM_HDR", "128",
   "Buffer size to reserve for active message headers. If set to 0, the transport will\n"
   "not support zero-copy active messages.",
   ucs_offsetof(uct_rc_verbs_iface_config_t, max_am_hdr), UCS_CONFIG_TYPE_MEMUNITS},

  {NULL}
};

static UCS_F_NOINLINE unsigned
uct_rc_verbs_iface_post_recv_always(uct_rc_verbs_iface_t *iface, unsigned max)
{
    struct ibv_recv_wr *bad_wr;
    uct_ib_recv_wr_t *wrs;
    unsigned count;
    int ret;

    wrs  = alloca(sizeof *wrs  * max);

    count = uct_ib_iface_prepare_rx_wrs(&iface->super.super,
                                        iface->super.rx.mp, wrs, max);
    if (count == 0) {
        return 0;
    }

    ret = ibv_post_srq_recv(iface->super.rx.srq, &wrs[0].ibwr, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_post_srq_recv() returned %d: %m", ret);
    }
    iface->super.rx.available -= count;

    return count;
}

static inline unsigned uct_rc_verbs_iface_post_recv(uct_rc_verbs_iface_t *iface,
                                                    int fill)
{
    unsigned batch = iface->super.config.rx_max_batch;
    unsigned count;

    if (iface->super.rx.available < batch) {
        if (!fill) {
            return 0;
        } else {
            count = iface->super.rx.available;
        }
    } else {
        count = batch;
    }

    return uct_rc_verbs_iface_post_recv_always(iface, count);
}

static inline void uct_rc_verbs_iface_poll_tx(uct_rc_verbs_iface_t *iface)
{
    uct_rc_verbs_ep_t *ep;
    struct ibv_wc wc[UCT_IB_MAX_WC];
    unsigned count;
    int i, ret;

    ret = ibv_poll_cq(iface->super.super.send_cq, UCT_IB_MAX_WC, wc);
    if (ret > 0) {
        for (i = 0; i < ret; ++i) {
            if (ucs_unlikely(wc[i].status != IBV_WC_SUCCESS)) {
                ucs_fatal("Send completion with error: %s", ibv_wc_status_str(wc[i].status));
            }

            ep = ucs_derived_of(uct_rc_iface_lookup_ep(&iface->super, wc[i].qp_num), uct_rc_verbs_ep_t);
            ucs_assert(ep != NULL);

            count = wc[i].wr_id + 1;
            ep->tx.available            += count;
            ep->tx.ci                   += count;
            iface->super.tx.outstanding -= count;

            ucs_callbackq_pull(&ep->super.tx.comp, ep->tx.ci);
        }
    } else if (ucs_unlikely(ret < 0)) {
        ucs_fatal("Failed to poll send CQ");
    }
}

static inline ucs_status_t uct_rc_verbs_iface_poll_rx(uct_rc_verbs_iface_t *iface)
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
            hdr = uct_ib_iface_recv_desc_hdr(&iface->super.super, desc);
            VALGRIND_MAKE_MEM_DEFINED(hdr, wc[i].byte_len);

            status = uct_rc_iface_invoke_am(&iface->super, hdr, wc[i].byte_len);
            if (status == UCS_OK) {
                ucs_mpool_put(desc);
            }
        }

        iface->super.rx.available += ret;
        return UCS_OK;
    } else if (ret == 0) {
        uct_rc_verbs_iface_post_recv(iface, 0);
        return UCS_ERR_NO_PROGRESS;
    } else {
        ucs_fatal("Failed to poll receive CQ");
    }
}

static void uct_rc_verbs_iface_progress(void *arg)
{
    uct_rc_verbs_iface_t *iface = arg;
    ucs_status_t status;

    status = uct_rc_verbs_iface_poll_rx(iface);
    if (status == UCS_ERR_NO_PROGRESS) {
        uct_rc_verbs_iface_poll_tx(iface);
    }
}

static ucs_status_t uct_rc_verbs_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_verbs_iface_t);
    struct ibv_qp_cap cap;
    struct ibv_qp *qp;
    ucs_status_t status;

    uct_rc_iface_query(&iface->super, iface_attr);

    /* Create a QP in order to find out capabilities */
    status = uct_rc_iface_qp_create(&iface->super, &qp, &cap);
    if (status != UCS_OK) {
        return status;
    }
    ibv_destroy_qp(qp);

    /* PUT */
    iface_attr->cap.put.max_short = cap.max_inline_data;
    iface_attr->cap.put.max_bcopy = iface->super.super.config.seg_size;
    iface_attr->cap.put.max_zcopy =
                        uct_ib_iface_port_attr(&iface->super.super)->max_msg_sz;

    /* AM */
    iface_attr->cap.am.max_short  = cap.max_inline_data
                                        - sizeof(uct_rc_hdr_t);
    iface_attr->cap.am.max_bcopy  = iface->super.super.config.seg_size
                                        - sizeof(uct_rc_hdr_t);

    if (iface->am_hdr_mp != NULL) {
        iface_attr->cap.am.max_zcopy  = iface->super.super.config.seg_size
                                            - sizeof(uct_rc_hdr_t);
        iface_attr->cap.am.max_hdr    = iface->config.max_am_hdr
                                            - sizeof(uct_rc_hdr_t);
    } else {
        iface_attr->cap.am.max_zcopy  = 0;
        iface_attr->cap.am.max_hdr    = 0;
    }

    /* TODO add get and atomics */
    return UCS_OK;
}

static void uct_rc_verbs_iface_amh_desc_init(uct_iface_h tl_iface, void *obj,
                                             uct_lkey_t lkey)
{
    uct_rc_iface_send_desc_t *desc = obj;

    desc->lkey = uct_ib_lkey_mr(lkey)->lkey;
    desc->queue.super.func = (void*)ucs_mpool_put;
}

static UCS_CLASS_INIT_FUNC(uct_rc_verbs_iface_t, uct_context_h context,
                           const char *dev_name, size_t rx_headroom,
                           uct_iface_config_t *tl_config)
{
    uct_rc_verbs_iface_config_t *config =
                    ucs_derived_of(tl_config, uct_rc_verbs_iface_config_t);
    ucs_status_t status;

    extern uct_iface_ops_t uct_rc_verbs_iface_ops;
    UCS_CLASS_CALL_SUPER_INIT(&uct_rc_verbs_iface_ops, context, dev_name,
                              rx_headroom, 0, &config->super);

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

    /* Configuration */
    self->config.max_am_hdr                 = config->max_am_hdr;

    /* Create AH headers mempool */
    if (self->config.max_am_hdr >= sizeof(uct_rc_hdr_t)) {
        status = uct_iface_mpool_create(&self->super.super.super.super,
                                        sizeof(uct_rc_iface_send_desc_t) + self->config.max_am_hdr,
                                        sizeof(uct_rc_iface_send_desc_t),
                                        UCS_SYS_CACHE_LINE_SIZE,
                                        &config->super.super.tx.mp,
                                        self->super.config.tx_qp_len,
                                        uct_rc_verbs_iface_amh_desc_init,
                                        "rc_verbs_am_hdr_desc", &self->am_hdr_mp);
        if (status != UCS_OK) {
            goto err;
        }
    } else {
        ucs_debug("header buffer is too small - disabling zero-copy active messages");
        self->am_hdr_mp = NULL;
    }

    while (self->super.rx.available > 0) {
        if (uct_rc_verbs_iface_post_recv(self, 1) == 0) {
            ucs_error("failed to post receives");
            status = UCS_ERR_NO_MEMORY;
            goto err_destroy_amh_mp;
        }
    }

    ucs_notifier_chain_add(&context->progress_chain, uct_rc_verbs_iface_progress,
                           self);
    return UCS_OK;

err_destroy_amh_mp:
    if (self->am_hdr_mp != NULL) {
        ucs_mpool_destroy(self->am_hdr_mp);
    }
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_verbs_iface_t)
{
    uct_context_h context = uct_ib_iface_device(&self->super.super)->super.context;

    if (self->am_hdr_mp != NULL) {
        ucs_mpool_destroy(self->am_hdr_mp);
    }
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
    .ep_am_bcopy         = uct_rc_verbs_ep_am_bcopy,
    .ep_am_zcopy         = uct_rc_verbs_ep_am_zcopy,
    .ep_put_short        = uct_rc_verbs_ep_put_short,
    .ep_put_bcopy        = uct_rc_verbs_ep_put_bcopy,
    .ep_put_zcopy        = uct_rc_verbs_ep_put_zcopy,
    .ep_flush            = uct_rc_verbs_ep_flush,
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
    uct_register_tl(context, "rc", uct_rc_verbs_iface_config_table,
                    sizeof(uct_rc_verbs_iface_config_t), "RC_VERBS_", &uct_rc_verbs_tl_ops);
}

UCS_COMPONENT_DEFINE(uct_context_t, rc_verbs, uct_rc_verbs_register, ucs_empty_function, 0)
