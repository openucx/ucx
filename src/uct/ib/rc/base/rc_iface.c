/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "rc_iface.h"
#include "rc_ep.h"

#include <ucs/arch/cpu.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/type/class.h>


ucs_config_field_t uct_rc_iface_config_table[] = {
  {"IB_", "RX_INLINE=64;RX_QUEUE_LEN=4095", NULL,
   ucs_offsetof(uct_rc_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_ib_iface_config_table)},

  {"PATH_MTU", "default",
   "Path MTU. \"default\" will select the best MTU for the device.",
   ucs_offsetof(uct_rc_iface_config_t, path_mtu), UCS_CONFIG_TYPE_ENUM(uct_ib_mtu_values)},

  {"MAX_RD_ATOMIC", "4",
   "Maximal number of outstanding read or atomic replies",
   ucs_offsetof(uct_rc_iface_config_t, max_rd_atomic), UCS_CONFIG_TYPE_UINT},

  {"TIMEOUT", "1.0s",
   "Transport timeout",
   ucs_offsetof(uct_rc_iface_config_t, tx.timeout), UCS_CONFIG_TYPE_TIME},

  {"RETRY_COUNT", "7",
   "Transport retries",
   ucs_offsetof(uct_rc_iface_config_t, tx.retry_count), UCS_CONFIG_TYPE_UINT},

  {"RNR_TIMEOUT", "30ms",
   "RNR timeout",
   ucs_offsetof(uct_rc_iface_config_t,tx. rnr_timeout), UCS_CONFIG_TYPE_TIME},

  {"RNR_RETRY_COUNT", "7",
   "RNR retries",
   ucs_offsetof(uct_rc_iface_config_t, tx.rnr_retry_count), UCS_CONFIG_TYPE_UINT},

  {"TX_CQ_LEN", "4096",
   "Length of send completion queue. This limits the total number of outstanding signaled sends.",
   ucs_offsetof(uct_rc_iface_config_t, tx.cq_len), UCS_CONFIG_TYPE_UINT},

  {"FC_ENABLE", "y",
   "Enable flow control protocol to prevent sender from overwhelming the receiver,\n"
   "thus avoiding RC RnR backoff timer.",
   ucs_offsetof(uct_rc_iface_config_t, fc.enable), UCS_CONFIG_TYPE_BOOL},

  {"FC_WND_SIZE", "512",
   "The size of flow control window per endpoint. limits the number of AM\n"
   "which can be sent w/o acknowledgment.",
   ucs_offsetof(uct_rc_iface_config_t, fc.wnd_size), UCS_CONFIG_TYPE_UINT},

  {"FC_HARD_THRESH", "0.25",
   "Threshold for sending hard request for FC credits to the peer. This value\n"
   "refers to the percentage of the FC_WND_SIZE value. (must be > 0 and < 1)",
   ucs_offsetof(uct_rc_iface_config_t, fc.hard_thresh), UCS_CONFIG_TYPE_DOUBLE},

#if HAVE_DECL_IBV_EXP_QP_OOO_RW_DATA_PLACEMENT
  {"OOO_RW", "n",
   "Enable out-of-order RDMA data placement",
   ucs_offsetof(uct_rc_iface_config_t, ooo_rw), UCS_CONFIG_TYPE_BOOL},
#endif

  {NULL}
};


ucs_config_field_t uct_rc_fc_config_table[] = {
  {"FC_SOFT_THRESH", "0.5",
   "Threshold for sending soft request for FC credits to the peer. This value\n"
   "refers to the percentage of the FC_WND_SIZE value. (must be > HARD_THRESH and < 1)",
   ucs_offsetof(uct_rc_fc_config_t, soft_thresh), UCS_CONFIG_TYPE_DOUBLE},

  {NULL}
};


#if ENABLE_STATS
static ucs_stats_class_t uct_rc_iface_stats_class = {
    .name = "rc_iface",
    .num_counters = UCT_RC_IFACE_STAT_LAST,
    .counter_names = {
        [UCT_RC_IFACE_STAT_RX_COMPLETION] = "rx_completion",
        [UCT_RC_IFACE_STAT_TX_COMPLETION] = "tx_completion",
        [UCT_RC_IFACE_STAT_NO_CQE]        = "no_cqe",
    }
};
#endif


static ucs_mpool_ops_t uct_rc_fc_pending_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
};


void uct_rc_iface_query(uct_rc_iface_t *iface, uct_iface_attr_t *iface_attr)
{
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super);

    uct_ib_iface_query(&iface->super,
                       ucs_max(sizeof(uct_rc_hdr_t), UCT_IB_RETH_LEN),
                       iface_attr);

    iface_attr->iface_addr_len  = 0;
    iface_attr->ep_addr_len     = sizeof(uct_rc_ep_address_t);
    iface_attr->cap.flags       = UCT_IFACE_FLAG_AM_SHORT |
                                  UCT_IFACE_FLAG_AM_BCOPY |
                                  UCT_IFACE_FLAG_AM_ZCOPY |
                                  UCT_IFACE_FLAG_PUT_SHORT |
                                  UCT_IFACE_FLAG_PUT_BCOPY |
                                  UCT_IFACE_FLAG_PUT_ZCOPY |
                                  UCT_IFACE_FLAG_GET_BCOPY |
                                  UCT_IFACE_FLAG_GET_ZCOPY |
                                  UCT_IFACE_FLAG_PENDING   |
                                  UCT_IFACE_FLAG_CONNECT_TO_EP |
                                  UCT_IFACE_FLAG_AM_CB_SYNC |
                                  UCT_IFACE_FLAG_EVENT_SEND_COMP |
                                  UCT_IFACE_FLAG_EVENT_RECV_AM;

    if (uct_ib_atomic_is_supported(dev, 0, sizeof(uint64_t))) {
        iface_attr->cap.flags  |= UCT_IFACE_FLAG_ATOMIC_ADD64 |
                                  UCT_IFACE_FLAG_ATOMIC_FADD64 |
                                  UCT_IFACE_FLAG_ATOMIC_CSWAP64 |
                                  UCT_IFACE_FLAG_ATOMIC_DEVICE;
    }

    if (uct_ib_atomic_is_supported(dev, 1, sizeof(uint64_t))) {
        iface_attr->cap.flags |= UCT_IFACE_FLAG_ATOMIC_SWAP64 |
                                 UCT_IFACE_FLAG_ATOMIC_DEVICE;
    }

    if (uct_ib_atomic_is_supported(dev, 1, sizeof(uint32_t))) {
        iface_attr->cap.flags |= UCT_IFACE_FLAG_ATOMIC_ADD32 |
                                 UCT_IFACE_FLAG_ATOMIC_FADD32 |
                                 UCT_IFACE_FLAG_ATOMIC_SWAP32 |
                                 UCT_IFACE_FLAG_ATOMIC_CSWAP32 |
                                 UCT_IFACE_FLAG_ATOMIC_DEVICE;
    }

    iface_attr->cap.put.opt_zcopy_align = UCS_SYS_PCI_MAX_PAYLOAD;
    iface_attr->cap.get.opt_zcopy_align = UCS_SYS_PCI_MAX_PAYLOAD;
    iface_attr->cap.am.opt_zcopy_align  = UCS_SYS_PCI_MAX_PAYLOAD;
    iface_attr->cap.put.align_mtu = uct_ib_mtu_value(iface->config.path_mtu);
    iface_attr->cap.get.align_mtu = uct_ib_mtu_value(iface->config.path_mtu);
    iface_attr->cap.am.align_mtu  = uct_ib_mtu_value(iface->config.path_mtu);
}

void uct_rc_iface_add_ep(uct_rc_iface_t *iface, uct_rc_ep_t *ep,
                         unsigned qp_num)
{
    uct_rc_ep_t ***ptr, **memb;

    ptr = &iface->eps[qp_num >> UCT_RC_QP_TABLE_ORDER];
    if (*ptr == NULL) {
        *ptr = ucs_calloc(UCS_BIT(UCT_RC_QP_TABLE_MEMB_ORDER), sizeof(**ptr),
                           "rc qp table");
    }

    memb = &(*ptr)[qp_num &  UCS_MASK(UCT_RC_QP_TABLE_MEMB_ORDER)];
    ucs_assert(*memb == NULL);
    *memb = ep;
}

void uct_rc_iface_remove_ep(uct_rc_iface_t *iface, unsigned qp_num)
{
    uct_rc_ep_t **memb;

    memb = &iface->eps[qp_num >> UCT_RC_QP_TABLE_ORDER]
                      [qp_num &  UCS_MASK(UCT_RC_QP_TABLE_MEMB_ORDER)];
    ucs_assert(*memb != NULL);
    *memb = NULL;
}

ucs_status_t uct_rc_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                uct_completion_t *comp)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);
    ucs_status_t status;
    unsigned count;
    uct_rc_ep_t *ep;

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    count = 0;
    ucs_list_for_each(ep, &iface->ep_list, list) {
        status = uct_ep_flush(&ep->super.super, 0, NULL);
        if ((status == UCS_ERR_NO_RESOURCE) || (status == UCS_INPROGRESS)) {
            ++count;
        } else if (status != UCS_OK) {
            return status;
        }
    }

    if (count != 0) {
        UCT_TL_IFACE_STAT_FLUSH_WAIT(&iface->super.super);
        return UCS_INPROGRESS;
    }

    UCT_TL_IFACE_STAT_FLUSH(&iface->super.super);
    return UCS_OK;
}

void uct_rc_iface_send_desc_init(uct_iface_h tl_iface, void *obj, uct_mem_h memh)
{
    uct_rc_iface_send_desc_t *desc = obj;
    uct_ib_mem_t *ib_memh = memh;

    desc->lkey = ib_memh->lkey;
    desc->super.flags = 0;
}

static void uct_rc_iface_set_path_mtu(uct_rc_iface_t *iface,
                                      const uct_rc_iface_config_t *config)
{
    enum ibv_mtu port_mtu = uct_ib_iface_port_attr(&iface->super)->active_mtu;
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super);

    /* MTU is set by user configuration */
    if (config->path_mtu != UCT_IB_MTU_DEFAULT) {
        iface->config.path_mtu = config->path_mtu + (IBV_MTU_512 - UCT_IB_MTU_512);
    } else if ((port_mtu > IBV_MTU_2048) && (dev->dev_attr.vendor_id == 0x02c9) &&
        ((dev->dev_attr.vendor_part_id == 4099) || (dev->dev_attr.vendor_part_id == 4100) ||
         (dev->dev_attr.vendor_part_id == 4103) || (dev->dev_attr.vendor_part_id == 4104)))
    {
        /* On some devices optimal path_mtu is 2048 */
        iface->config.path_mtu = IBV_MTU_2048;
    } else {
        iface->config.path_mtu = port_mtu;
    }
}

ucs_status_t uct_rc_init_fc_thresh(uct_rc_fc_config_t *fc_cfg,
                                   uct_rc_iface_config_t *rc_cfg,
                                   uct_rc_iface_t *iface)
{
    /* Check FC parameters correctness */
    if ((fc_cfg->soft_thresh <= rc_cfg->fc.hard_thresh) ||
        (fc_cfg->soft_thresh >= 1)) {
        ucs_error("The factor for soft FC threshold should be bigger"
                  " than FC_HARD_THRESH value and less than 1 (s=%f, h=%f)",
                  fc_cfg->soft_thresh, rc_cfg->fc.hard_thresh);
        return UCS_ERR_INVALID_PARAM;
    }

    if (rc_cfg->fc.enable) {
        iface->config.fc_soft_thresh = ucs_max((int)(iface->config.fc_wnd_size *
                                               fc_cfg->soft_thresh), 1);
    } else {
        iface->config.fc_soft_thresh  = 0;
    }
    return UCS_OK;
}

ucs_status_t uct_rc_iface_fc_handler(uct_rc_iface_t *iface, unsigned qp_num,
                                     uct_rc_hdr_t *hdr, unsigned length,
                                     uint32_t imm_data, uint16_t lid, unsigned flags)
{
    ucs_status_t status;
    int16_t      cur_wnd;
    uct_rc_fc_request_t *fc_req;
    uct_rc_ep_t  *ep  = uct_rc_iface_lookup_ep(iface, qp_num);
    uint8_t fc_hdr    = uct_rc_fc_get_fc_hdr(hdr->am_id);

    ucs_assert(iface->config.fc_enabled);

    if (fc_hdr & UCT_RC_EP_FC_FLAG_GRANT) {
        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_RX_GRANT, 1);

        /* Got either grant flag or special FC grant message */
        cur_wnd = ep->fc.fc_wnd;

        /* Peer granted resources, so update wnd */
        ep->fc.fc_wnd = iface->config.fc_wnd_size;
        UCS_STATS_SET_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_FC_WND, ep->fc.fc_wnd);

        /* To preserve ordering we have to dispatch all pending
         * operations if current fc_wnd is <= 0
         * (otherwise it will be dispatched by tx progress) */
        if (cur_wnd <= 0) {
            ucs_arbiter_group_schedule(&iface->tx.arbiter, &ep->arb_group);
            ucs_arbiter_dispatch(&iface->tx.arbiter, 1,
                                 uct_rc_ep_process_pending, NULL);
        }
        if  (fc_hdr == UCT_RC_EP_FC_PURE_GRANT) {
            /* Special FC grant message can't be bundled with any other FC
             * request. Stop processing this AM and do not invoke AM handler */
            UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_RX_PURE_GRANT, 1);
            return UCS_OK;
        }
    }

    if (fc_hdr & UCT_RC_EP_FC_FLAG_SOFT_REQ) {
        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_RX_SOFT_REQ, 1);

        /* Got soft credit request. Mark ep that it needs to grant
         * credits to the peer in outgoing AM (if any). */
        ep->fc.flags |= UCT_RC_EP_FC_FLAG_GRANT;

    } else if (fc_hdr & UCT_RC_EP_FC_FLAG_HARD_REQ) {
        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_RX_HARD_REQ, 1);
        fc_req = ucs_mpool_get(&iface->tx.fc_mp);
        if (ucs_unlikely(fc_req == NULL)) {
            ucs_error("Failed to allocate FC request. "
                      "Grant will not be sent on ep %p", ep);
            return UCS_ERR_NO_MEMORY;
        }
        fc_req->ep         = &ep->super.super;
        fc_req->super.func = uct_rc_ep_fc_grant;

        /* Got hard credit request. Send grant to the peer immediately */
        status = uct_rc_ep_fc_grant(&fc_req->super);

        if (status == UCS_ERR_NO_RESOURCE){
            status = uct_ep_pending_add(&ep->super.super, &fc_req->super);
        }
        ucs_assertv_always(status == UCS_OK, "Failed to send FC grant msg: %s",
                           ucs_status_string(status));
    }

    return uct_iface_invoke_am(&iface->super.super,
                               (hdr->am_id & ~UCT_RC_EP_FC_MASK),
                               hdr + 1, length, flags);
}

static ucs_status_t uct_rc_iface_tx_ops_init(uct_rc_iface_t *iface)
{
    const unsigned count = iface->config.tx_ops_count;
    uct_rc_iface_send_op_t *op;

    iface->tx.ops_buffer = ucs_calloc(count, sizeof(*iface->tx.ops_buffer),
                                      "rc_tx_ops");
    if (iface->tx.ops_buffer == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    iface->tx.free_ops = &iface->tx.ops_buffer[0];
    for (op = iface->tx.ops_buffer; op < iface->tx.ops_buffer + count - 1; ++op) {
        op->handler = uct_rc_ep_send_op_completion_handler;
        op->flags   = UCT_RC_IFACE_SEND_OP_FLAG_IFACE;
        op->iface   = iface;
        op->next    = op + 1;
    }
    iface->tx.ops_buffer[count - 1].next = NULL;
    return UCS_OK;
}

static void uct_rc_iface_tx_ops_cleanup(uct_rc_iface_t *iface)
{
    const unsigned total_count = iface->config.tx_ops_count;
    uct_rc_iface_send_op_t *op;
    unsigned free_count;

    free_count = 0;
    for (op = iface->tx.free_ops; op != NULL; op = op->next) {
        ++free_count;
        ucs_assert(free_count <= total_count);
    }
    if (free_count != iface->config.tx_ops_count) {
        ucs_warn("rc_iface %p: %u/%d send ops were not released", iface,
                 total_count- free_count, total_count);
    }
    ucs_free(iface->tx.ops_buffer);
}

UCS_CLASS_INIT_FUNC(uct_rc_iface_t, uct_rc_iface_ops_t *ops, uct_md_h md,
                    uct_worker_h worker, const uct_iface_params_t *params,
                    const uct_rc_iface_config_t *config, unsigned rx_priv_len,
                    unsigned rx_cq_len, unsigned rx_hdr_len,
                    unsigned srq_size, unsigned fc_req_size)
{
    struct ibv_srq_init_attr srq_init_attr;
    ucs_status_t status;
    uct_ib_device_t *dev = &ucs_derived_of(md, uct_ib_md_t)->dev;
    unsigned tx_cq_len = config->tx.cq_len;

    UCS_CLASS_CALL_SUPER_INIT(uct_ib_iface_t, &ops->super, md, worker, params,
                              rx_priv_len, rx_hdr_len, tx_cq_len, rx_cq_len,
                              SIZE_MAX, &config->super);

    self->tx.cq_available           = tx_cq_len - 1; /* Reserve one for error */
    self->rx.srq.available          = 0;
    self->config.tx_qp_len          = config->super.tx.queue_len;
    self->config.tx_min_sge         = config->super.tx.min_sge;
    self->config.tx_min_inline      = config->super.tx.min_inline;
    self->config.tx_moderation      = ucs_min(config->super.tx.cq_moderation,
                                              config->super.tx.queue_len / 4);
    self->config.tx_ops_count       = tx_cq_len;
    self->config.rx_inline          = config->super.rx.inl;
    self->config.min_rnr_timer      = uct_ib_to_fabric_time(config->tx.rnr_timeout);
    self->config.timeout            = uct_ib_to_fabric_time(config->tx.timeout);
    self->config.rnr_retry          = ucs_min(config->tx.rnr_retry_count,
                                              UCR_RC_QP_MAX_RETRY_COUNT);
    self->config.retry_cnt          = ucs_min(config->tx.retry_count,
                                              UCR_RC_QP_MAX_RETRY_COUNT);
    self->config.max_rd_atomic      = config->max_rd_atomic;
    self->config.ooo_rw             = config->ooo_rw;

    uct_rc_iface_set_path_mtu(self, config);
    memset(self->eps, 0, sizeof(self->eps));
    ucs_arbiter_init(&self->tx.arbiter);
    ucs_list_head_init(&self->ep_list);

    /* Check FC parameters correctness */
    if ((config->fc.hard_thresh <= 0) || (config->fc.hard_thresh >= 1)) {
        ucs_error("The factor for hard FC threshold should be > 0 and < 1 (%f)",
                  config->fc.hard_thresh);
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    /* Create RX buffers mempool */
    status = uct_ib_iface_recv_mpool_init(&self->super, &config->super,
                                          "rc_recv_desc", &self->rx.mp);
    if (status != UCS_OK) {
        goto err;
    }

    /* Create TX buffers mempool */
    status = uct_iface_mpool_init(&self->super.super,
                                  &self->tx.mp,
                                  sizeof(uct_rc_iface_send_desc_t) + self->super.config.seg_size,
                                  sizeof(uct_rc_iface_send_desc_t),
                                  UCS_SYS_CACHE_LINE_SIZE,
                                  &config->super.tx.mp,
                                  self->config.tx_qp_len,
                                  uct_rc_iface_send_desc_init,
                                  "rc_send_desc");
    if (status != UCS_OK) {
        goto err_destroy_rx_mp;
    }

    /* Allocate tx operations */
    status = uct_rc_iface_tx_ops_init(self);
    if (status != UCS_OK) {
        goto err_destroy_tx_mp;
    }

    /* Create SRQ */
    if (srq_size > 0) {
        srq_init_attr.attr.max_sge   = 1;
        srq_init_attr.attr.max_wr    = srq_size;
        srq_init_attr.attr.srq_limit = 0;
        srq_init_attr.srq_context    = self;
        self->rx.srq.srq = ibv_create_srq(uct_ib_iface_md(&self->super)->pd, &srq_init_attr);
        if (self->rx.srq.srq == NULL) {
            ucs_error("failed to create SRQ: %m");
            status = UCS_ERR_IO_ERROR;
            goto err_free_tx_ops;
        }
        self->rx.srq.available       = srq_init_attr.attr.max_wr;
    } else {
        self->rx.srq.srq             = NULL;
    }

    /* Set atomic handlers according to atomic reply endianness */
    self->config.atomic64_handler = uct_ib_atomic_is_be_reply(dev, 0, sizeof(uint64_t)) ?
                                    uct_rc_ep_atomic_handler_64_be1 :
                                    uct_rc_ep_atomic_handler_64_be0;
    self->config.atomic32_ext_handler = uct_ib_atomic_is_be_reply(dev, 1, sizeof(uint32_t)) ?
                                        uct_rc_ep_atomic_handler_32_be1 :
                                        uct_rc_ep_atomic_handler_32_be0;
    self->config.atomic64_ext_handler = uct_ib_atomic_is_be_reply(dev, 1, sizeof(uint64_t)) ?
                                        uct_rc_ep_atomic_handler_64_be1 :
                                        uct_rc_ep_atomic_handler_64_be0;

    status = UCS_STATS_NODE_ALLOC(&self->stats, &uct_rc_iface_stats_class,
                                  self->super.super.stats);
    if (status != UCS_OK) {
        goto err_destroy_srq;
    }

    self->config.fc_enabled      = config->fc.enable;

    if (self->config.fc_enabled) {
        /* Assume that number of recv buffers is the same on all peers.
         * Then FC window size is the same for all endpoints as well.
         * TODO: Make wnd size to be a property of the particular interface.
         * We could distribute it via rc address then.*/
        self->config.fc_wnd_size     = ucs_min(config->fc.wnd_size,
                                               config->super.rx.queue_len);
        self->config.fc_hard_thresh  = ucs_max((int)(self->config.fc_wnd_size *
                                               config->fc.hard_thresh), 1);

        /* Create mempool for pending requests for FC grant */
        status = ucs_mpool_init(&self->tx.fc_mp,
                                0,
                                fc_req_size,
                                0,
                                1,
                                128,
                                UINT_MAX,
                                &uct_rc_fc_pending_mpool_ops,
                                "pending-fc-grants-only");
        if (status != UCS_OK) {
            goto err_destroy_stats;
        }
    } else {
        self->config.fc_wnd_size     = INT16_MAX;
        self->config.fc_hard_thresh  = 0;
    }

    return UCS_OK;

err_destroy_stats:
    UCS_STATS_NODE_FREE(self->stats);
err_destroy_srq:
    if (self->rx.srq.srq != NULL) {
        ibv_destroy_srq(self->rx.srq.srq);
    }
err_free_tx_ops:
    uct_rc_iface_tx_ops_cleanup(self);
err_destroy_tx_mp:
    ucs_mpool_cleanup(&self->tx.mp, 1);
err_destroy_rx_mp:
    ucs_mpool_cleanup(&self->rx.mp, 1);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_iface_t)
{
    unsigned i;
    int ret;

    /* Release table. TODO release on-demand when removing ep. */
    for (i = 0; i < UCT_RC_QP_TABLE_SIZE; ++i) {
        ucs_free(self->eps[i]);
    }

    if (!ucs_list_is_empty(&self->ep_list)) {
        ucs_warn("some eps were not destroyed");
    }

    ucs_arbiter_cleanup(&self->tx.arbiter);

    UCS_STATS_NODE_FREE(self->stats);

    if (self->rx.srq.srq != NULL) {
        /* TODO flush RX buffers */
        ret = ibv_destroy_srq(self->rx.srq.srq);
        if (ret) {
            ucs_warn("failed to destroy SRQ: %m");
        }
    }
    uct_rc_iface_tx_ops_cleanup(self);
    ucs_mpool_cleanup(&self->tx.mp, 1);
    ucs_mpool_cleanup(&self->rx.mp, 0); /* Cannot flush SRQ */
    if (self->config.fc_enabled) {
        ucs_mpool_cleanup(&self->tx.fc_mp, 1);
    }
}

UCS_CLASS_DEFINE(uct_rc_iface_t, uct_ib_iface_t);


ucs_status_t uct_rc_iface_qp_create(uct_rc_iface_t *iface, int qp_type,
                                    struct ibv_qp **qp_p, struct ibv_qp_cap *cap,
                                    struct ibv_srq *srq, unsigned max_send_wr)
{
    uct_ib_device_t *dev UCS_V_UNUSED = uct_ib_iface_device(&iface->super);
    struct ibv_exp_qp_init_attr qp_init_attr;
    const char *qp_type_str;
    struct ibv_qp *qp;
    int inline_recv = 0;

    /* Check QP type */
    if (qp_type == IBV_QPT_RC) {
        qp_type_str = "rc";
#if HAVE_DECL_IBV_EXP_QPT_DC_INI
    } else if (qp_type == IBV_EXP_QPT_DC_INI) {
        qp_type_str = "dci";
#endif
    } else {
        ucs_bug("invalid qp type: %d", qp_type);
    }

    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.qp_context          = NULL;
    qp_init_attr.send_cq             = iface->super.send_cq;
    qp_init_attr.recv_cq             = iface->super.recv_cq;
    if (qp_type == IBV_QPT_RC) {
        qp_init_attr.srq             = srq;
    }
    qp_init_attr.cap.max_send_wr     = max_send_wr;
    qp_init_attr.cap.max_recv_wr     = 0;
    qp_init_attr.cap.max_send_sge    = iface->config.tx_min_sge;
    qp_init_attr.cap.max_recv_sge    = 1;
    qp_init_attr.cap.max_inline_data = iface->config.tx_min_inline;
    qp_init_attr.qp_type             = qp_type;
    qp_init_attr.sq_sig_all          = !iface->config.tx_moderation;

#if HAVE_DECL_IBV_EXP_CREATE_QP
    qp_init_attr.comp_mask           = IBV_EXP_QP_INIT_ATTR_PD;
    qp_init_attr.pd                  = uct_ib_iface_md(&iface->super)->pd;

#  if HAVE_IB_EXT_ATOMICS
    qp_init_attr.comp_mask          |= IBV_EXP_QP_INIT_ATTR_ATOMICS_ARG;
    qp_init_attr.max_atomic_arg      = UCT_RC_MAX_ATOMIC_SIZE;
#  endif

#  if HAVE_DECL_IBV_EXP_ATOMIC_HCA_REPLY_BE
    if (dev->dev_attr.exp_atomic_cap == IBV_EXP_ATOMIC_HCA_REPLY_BE) {
        qp_init_attr.comp_mask       |= IBV_EXP_QP_INIT_ATTR_CREATE_FLAGS;
        qp_init_attr.exp_create_flags = IBV_EXP_QP_CREATE_ATOMIC_BE_REPLY;
    }
#  endif

#  if HAVE_STRUCT_IBV_EXP_QP_INIT_ATTR_MAX_INL_RECV
    qp_init_attr.comp_mask           |= IBV_EXP_QP_INIT_ATTR_INL_RECV;
    qp_init_attr.max_inl_recv         = iface->config.rx_inline;
#  endif

#  if HAVE_EXP_UMR
#    if (HAVE_IBV_EXP_QP_CREATE_UMR_CAPS || HAVE_EXP_UMR_NEW_API)
    qp_init_attr.comp_mask           |= IBV_EXP_QP_INIT_ATTR_MAX_INL_KLMS;
    qp_init_attr.max_inl_send_klms    = dev->dev_attr.umr_caps.max_send_wqe_inline_klms;
#    else
    qp_init_attr.comp_mask           |= IBV_EXP_QP_INIT_ATTR_MAX_INL_KLMS;
    qp_init_attr.max_inl_send_klms    = dev->dev_attr.max_send_wqe_inline_klms;
#    endif

#    if HAVE_IBV_EXP_QP_CREATE_UMR
    qp_init_attr.comp_mask           |= IBV_EXP_QP_INIT_ATTR_CREATE_FLAGS;
    qp_init_attr.exp_create_flags    |= IBV_EXP_QP_CREATE_UMR;
#    endif
#  endif

    qp = ibv_exp_create_qp(dev->ibv_context, &qp_init_attr);
#else
    qp = ibv_create_qp(uct_ib_iface_md(&iface->super)->pd, &qp_init_attr);
#endif
    if (qp == NULL) {
        ucs_error("failed to create qp: %m");
        return UCS_ERR_IO_ERROR;
    }

#if HAVE_STRUCT_IBV_EXP_QP_INIT_ATTR_MAX_INL_RECV
    qp_init_attr.max_inl_recv = qp_init_attr.max_inl_recv / 2; /* Driver bug W/A */
    inline_recv = qp_init_attr.max_inl_recv;
#endif

    ucs_debug("created %s qp 0x%x tx %d rx %d tx_inline %d rx_inline %d",
              qp_type_str, qp->qp_num, qp_init_attr.cap.max_send_wr,
              qp_init_attr.cap.max_recv_wr, qp_init_attr.cap.max_inline_data,
              inline_recv);

    *qp_p = qp;
    *cap  = qp_init_attr.cap;
    return UCS_OK;
}

ucs_status_t uct_rc_iface_qp_init(uct_rc_iface_t *iface, struct ibv_qp *qp)
{
    struct ibv_qp_attr qp_attr;
    int ret;

    memset(&qp_attr, 0, sizeof(qp_attr));

    qp_attr.qp_state              = IBV_QPS_INIT;
    qp_attr.pkey_index            = iface->super.pkey_index;
    qp_attr.port_num              = iface->super.config.port_num;
    qp_attr.qp_access_flags       = IBV_ACCESS_LOCAL_WRITE   |
                                    IBV_ACCESS_REMOTE_WRITE  |
                                    IBV_ACCESS_REMOTE_READ   |
                                    IBV_ACCESS_REMOTE_ATOMIC |
                                    IBV_ACCESS_MW_BIND;
    ret = ibv_modify_qp(qp, &qp_attr,
                        IBV_QP_STATE      |
                        IBV_QP_PKEY_INDEX |
                        IBV_QP_PORT       |
                        IBV_QP_ACCESS_FLAGS);
    if (ret) {
         ucs_error("error modifying QP to INIT: %m");
         return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

ucs_status_t uct_rc_iface_qp_connect(uct_rc_iface_t *iface, struct ibv_qp *qp,
                                     const uint32_t dest_qp_num,
                                     struct ibv_ah_attr *ah_attr)
{
#if HAVE_DECL_IBV_EXP_QP_OOO_RW_DATA_PLACEMENT
    struct ibv_exp_qp_attr qp_attr;
#else
    struct ibv_qp_attr qp_attr;
#endif
    long qp_attr_mask;
    int ret;

    memset(&qp_attr, 0, sizeof(qp_attr));

    qp_attr.qp_state              = IBV_QPS_RTR;
    qp_attr.dest_qp_num           = dest_qp_num;
    qp_attr.rq_psn                = 0;
    qp_attr.path_mtu              = iface->config.path_mtu;
    qp_attr.max_dest_rd_atomic    = iface->config.max_rd_atomic;
    qp_attr.min_rnr_timer         = iface->config.min_rnr_timer;
    qp_attr.ah_attr               = *ah_attr;
    qp_attr_mask                  = IBV_QP_STATE              |
                                    IBV_QP_AV                 |
                                    IBV_QP_PATH_MTU           |
                                    IBV_QP_DEST_QPN           |
                                    IBV_QP_RQ_PSN             |
                                    IBV_QP_MAX_DEST_RD_ATOMIC |
                                    IBV_QP_MIN_RNR_TIMER;

#if HAVE_DECL_IBV_EXP_QP_OOO_RW_DATA_PLACEMENT
    if (iface->config.ooo_rw &&
        UCX_IB_DEV_IS_OOO_SUPPORTED(&uct_ib_iface_device(&iface->super)->dev_attr,
                                    rc)) {
        ucs_debug("enabling out-of-order on RC QP %x dev %s", qp->qp_num,
                   uct_ib_device_name(uct_ib_iface_device(&iface->super)));
        qp_attr_mask |= IBV_EXP_QP_OOO_RW_DATA_PLACEMENT;
    }
    ret = ibv_exp_modify_qp(qp, &qp_attr, qp_attr_mask);
#else
    ret = ibv_modify_qp(qp, &qp_attr, qp_attr_mask);
#endif
    if (ret) {
        ucs_error("error modifying QP to RTR: %m");
        return UCS_ERR_IO_ERROR;
    }

    qp_attr.qp_state              = IBV_QPS_RTS;
    qp_attr.sq_psn                = 0;
    qp_attr.timeout               = iface->config.timeout;
    qp_attr.rnr_retry             = iface->config.rnr_retry;
    qp_attr.retry_cnt             = iface->config.retry_cnt;
    qp_attr.max_rd_atomic         = iface->config.max_rd_atomic;
    qp_attr_mask                  = IBV_QP_STATE              |
                                    IBV_QP_TIMEOUT            |
                                    IBV_QP_RETRY_CNT          |
                                    IBV_QP_RNR_RETRY          |
                                    IBV_QP_SQ_PSN             |
                                    IBV_QP_MAX_QP_RD_ATOMIC;

#if HAVE_DECL_IBV_EXP_QP_OOO_RW_DATA_PLACEMENT
    ret = ibv_exp_modify_qp(qp, &qp_attr, qp_attr_mask);
#else
    ret = ibv_modify_qp(qp, &qp_attr, qp_attr_mask);
#endif
    if (ret) {
        ucs_error("error modifying QP to RTS: %m");
        return UCS_ERR_IO_ERROR;
    }

    ucs_debug("connected rc qp 0x%x on "UCT_IB_IFACE_FMT" to lid %d(+%d) sl %d "
              "remote_qp 0x%x mtu %zu timer %dx%d rnr %dx%d rd_atom %d",
              qp->qp_num, UCT_IB_IFACE_ARG(&iface->super), ah_attr->dlid,
              ah_attr->src_path_bits, ah_attr->sl, qp_attr.dest_qp_num,
              uct_ib_mtu_value(qp_attr.path_mtu), qp_attr.timeout,
              qp_attr.retry_cnt, qp_attr.min_rnr_timer, qp_attr.rnr_retry,
              qp_attr.max_rd_atomic);

    return UCS_OK;
}
