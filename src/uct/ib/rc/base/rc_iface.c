/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rc_iface.h"
#include "rc_ep.h"

#include <ucs/arch/cpu.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/type/class.h>


static const char *uct_rc_fence_mode_values[] = {
    [UCT_RC_FENCE_MODE_NONE]   = "none",
    [UCT_RC_FENCE_MODE_WEAK]   = "weak",
    [UCT_RC_FENCE_MODE_AUTO]   = "auto",
    [UCT_RC_FENCE_MODE_LAST]   = NULL
};

ucs_config_field_t uct_rc_iface_common_config_table[] = {
  {UCT_IB_CONFIG_PREFIX, "RX_INLINE=64;TX_INLINE_RESP=64;RX_QUEUE_LEN=4095;SEG_SIZE=8256", NULL,
   ucs_offsetof(uct_rc_iface_common_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_ib_iface_config_table)},

  {"MAX_RD_ATOMIC", "4",
   "Maximal number of outstanding read or atomic replies",
   ucs_offsetof(uct_rc_iface_common_config_t, max_rd_atomic), UCS_CONFIG_TYPE_UINT},

  {"TIMEOUT", "1.0s",
   "Transport timeout",
   ucs_offsetof(uct_rc_iface_common_config_t, tx.timeout), UCS_CONFIG_TYPE_TIME},

  {"RETRY_COUNT", "7",
   "Transport retries",
   ucs_offsetof(uct_rc_iface_common_config_t, tx.retry_count), UCS_CONFIG_TYPE_UINT},

  {"RNR_TIMEOUT", "1ms",
   "RNR timeout",
   ucs_offsetof(uct_rc_iface_common_config_t, tx.rnr_timeout), UCS_CONFIG_TYPE_TIME},

  {"RNR_RETRY_COUNT", "7",
   "RNR retries",
   ucs_offsetof(uct_rc_iface_common_config_t, tx.rnr_retry_count), UCS_CONFIG_TYPE_UINT},

  {"FC_ENABLE", "y",
   "Enable flow control protocol to prevent sender from overwhelming the receiver,\n"
   "thus avoiding RC RnR backoff timer.",
   ucs_offsetof(uct_rc_iface_common_config_t, fc.enable), UCS_CONFIG_TYPE_BOOL},

  {"FC_WND_SIZE", "512",
   "The size of flow control window per endpoint. limits the number of AM\n"
   "which can be sent w/o acknowledgment.",
   ucs_offsetof(uct_rc_iface_common_config_t, fc.wnd_size), UCS_CONFIG_TYPE_UINT},

  {"FC_HARD_THRESH", "0.25",
   "Threshold for sending hard request for FC credits to the peer. This value\n"
   "refers to the percentage of the FC_WND_SIZE value. (must be > 0 and < 1)",
   ucs_offsetof(uct_rc_iface_common_config_t, fc.hard_thresh), UCS_CONFIG_TYPE_DOUBLE},

#if HAVE_DECL_IBV_EXP_QP_OOO_RW_DATA_PLACEMENT
  {"OOO_RW", "n",
   "Enable out-of-order RDMA data placement",
   ucs_offsetof(uct_rc_iface_common_config_t, ooo_rw), UCS_CONFIG_TYPE_BOOL},
#endif

  {"FENCE", "auto",
   "IB fence type when API fence requested:\n"
   "  none   - fence is a no-op\n"
   "  weak   - fence makes sure remote reads are ordered with respect to remote writes\n"
   "  auto   - select fence mode based on hardware capabilities",
   ucs_offsetof(uct_rc_iface_common_config_t, fence_mode),
                UCS_CONFIG_TYPE_ENUM(uct_rc_fence_mode_values)},

  {"TX_NUM_GET_OPS", "",
   "The configuration parameter replaced by UCX_RC_TX_NUM_GET_BYTES.",
   UCS_CONFIG_DEPRECATED_FIELD_OFFSET, UCS_CONFIG_TYPE_DEPRECATED},

  {"MAX_GET_ZCOPY", "auto",
   "Maximal size of get operation with zcopy protocol.",
   ucs_offsetof(uct_rc_iface_common_config_t, tx.max_get_zcopy), UCS_CONFIG_TYPE_MEMUNITS},

  {"TX_NUM_GET_BYTES", "inf",
   "Maximal number of bytes simultaneously transferred by get/RDMA_READ operations.",
   ucs_offsetof(uct_rc_iface_common_config_t, tx.max_get_bytes), UCS_CONFIG_TYPE_MEMUNITS},

  {"TX_POLL_ALWAYS", "n",
   "When enabled, TX completions are polled every time the progress function is invoked.\n"
   "Otherwise poll TX completions only if no RX completions found.",
   ucs_offsetof(uct_rc_iface_common_config_t, tx.poll_always), UCS_CONFIG_TYPE_BOOL},

  {NULL}
};


/* Config relevant for rc_mlx5 and rc_verbs only (not for dc) */
ucs_config_field_t uct_rc_iface_config_table[] = {
  {"RC_", "MAX_NUM_EPS=256", NULL,
   ucs_offsetof(uct_rc_iface_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_rc_iface_common_config_table)},

  {"FC_SOFT_THRESH", "0.5",
   "Threshold for sending soft request for FC credits to the peer. This value\n"
   "refers to the percentage of the FC_WND_SIZE value. (must be > HARD_THRESH and < 1)",
   ucs_offsetof(uct_rc_iface_config_t, soft_thresh), UCS_CONFIG_TYPE_DOUBLE},

  {"TX_CQ_MODERATION", "64",
   "Maximum number of send WQEs which can be posted without requesting a completion.",
   ucs_offsetof(uct_rc_iface_config_t, tx_cq_moderation), UCS_CONFIG_TYPE_UINT},

  {"TX_CQ_LEN", "4096",
   "Length of send completion queue. This limits the total number of outstanding signaled sends.",
   ucs_offsetof(uct_rc_iface_config_t, tx_cq_len), UCS_CONFIG_TYPE_UINT},

  {NULL}
};


#ifdef ENABLE_STATS
static ucs_stats_class_t uct_rc_iface_stats_class = {
    .name = "rc_iface",
    .num_counters = UCT_RC_IFACE_STAT_LAST,
    .counter_names = {
        [UCT_RC_IFACE_STAT_RX_COMPLETION] = "rx_completion",
        [UCT_RC_IFACE_STAT_TX_COMPLETION] = "tx_completion",
        [UCT_RC_IFACE_STAT_NO_CQE]        = "no_cqe",
        [UCT_RC_IFACE_STAT_NO_READS]      = "no_reads"
    }
};

#endif /* ENABLE_STATS */


static ucs_mpool_ops_t uct_rc_pending_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
};

static ucs_mpool_ops_t uct_rc_send_op_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
};

ucs_status_t uct_rc_iface_query(uct_rc_iface_t *iface,
                                uct_iface_attr_t *iface_attr,
                                size_t put_max_short, size_t max_inline,
                                size_t am_max_hdr, size_t am_max_iov,
                                size_t am_min_hdr, size_t rma_max_iov)
{
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super);
    ucs_status_t status;

    status = uct_ib_iface_query(&iface->super,
                                ucs_max(sizeof(uct_rc_hdr_t), UCT_IB_RETH_LEN),
                                iface_attr);
    if (status != UCS_OK) {
        return status;
    }

    iface_attr->iface_addr_len  = 0;
    iface_attr->max_conn_priv   = 0;
    iface_attr->cap.flags       = UCT_IFACE_FLAG_AM_BCOPY        |
                                  UCT_IFACE_FLAG_AM_ZCOPY        |
                                  UCT_IFACE_FLAG_PUT_BCOPY       |
                                  UCT_IFACE_FLAG_PUT_ZCOPY       |
                                  UCT_IFACE_FLAG_GET_BCOPY       |
                                  UCT_IFACE_FLAG_GET_ZCOPY       |
                                  UCT_IFACE_FLAG_PENDING         |
                                  UCT_IFACE_FLAG_CONNECT_TO_EP   |
                                  UCT_IFACE_FLAG_CB_SYNC;
    iface_attr->cap.event_flags = UCT_IFACE_FLAG_EVENT_SEND_COMP |
                                  UCT_IFACE_FLAG_EVENT_RECV      |
                                  UCT_IFACE_FLAG_EVENT_FD;

    if (uct_ib_device_has_pci_atomics(dev)) {
        if (dev->pci_fadd_arg_sizes & sizeof(uint64_t)) {
            iface_attr->cap.atomic64.op_flags  |= UCS_BIT(UCT_ATOMIC_OP_ADD);
            iface_attr->cap.atomic64.fop_flags |= UCS_BIT(UCT_ATOMIC_OP_ADD);
        }
        if (dev->pci_cswap_arg_sizes & sizeof(uint64_t)) {
            iface_attr->cap.atomic64.fop_flags |= UCS_BIT(UCT_ATOMIC_OP_CSWAP);
        }
        iface_attr->cap.flags                  |= UCT_IFACE_FLAG_ATOMIC_CPU;
    } else {
        if (dev->atomic_arg_sizes & sizeof(uint64_t)) {
            /* TODO: remove deprecated flags */
            iface_attr->cap.flags              |= UCT_IFACE_FLAG_ATOMIC_DEVICE;

            iface_attr->cap.atomic64.op_flags  |= UCS_BIT(UCT_ATOMIC_OP_ADD);
            iface_attr->cap.atomic64.fop_flags |= UCS_BIT(UCT_ATOMIC_OP_ADD)  |
                                                  UCS_BIT(UCT_ATOMIC_OP_CSWAP);
        }
    }

    iface_attr->cap.put.opt_zcopy_align = UCS_SYS_PCI_MAX_PAYLOAD;
    iface_attr->cap.get.opt_zcopy_align = UCS_SYS_PCI_MAX_PAYLOAD;
    iface_attr->cap.am.opt_zcopy_align  = UCS_SYS_PCI_MAX_PAYLOAD;
    iface_attr->cap.put.align_mtu = uct_ib_mtu_value(iface->super.config.path_mtu);
    iface_attr->cap.get.align_mtu = uct_ib_mtu_value(iface->super.config.path_mtu);
    iface_attr->cap.am.align_mtu  = uct_ib_mtu_value(iface->super.config.path_mtu);


    /* PUT */
    iface_attr->cap.put.max_short = put_max_short;
    iface_attr->cap.put.max_bcopy = iface->super.config.seg_size;
    iface_attr->cap.put.min_zcopy = 0;
    iface_attr->cap.put.max_zcopy = uct_ib_iface_port_attr(&iface->super)->max_msg_sz;
    iface_attr->cap.put.max_iov   = rma_max_iov;

    /* GET */
    iface_attr->cap.get.max_bcopy = iface->super.config.seg_size;
    iface_attr->cap.get.min_zcopy = iface->super.config.max_inl_cqe[UCT_IB_DIR_TX] + 1;
    iface_attr->cap.get.max_zcopy = iface->config.max_get_zcopy;
    iface_attr->cap.get.max_iov   = rma_max_iov;

    /* AM */
    iface_attr->cap.am.max_short  = uct_ib_iface_hdr_size(max_inline, am_min_hdr);
    iface_attr->cap.am.max_bcopy  = iface->super.config.seg_size - am_min_hdr;
    iface_attr->cap.am.min_zcopy  = 0;
    iface_attr->cap.am.max_zcopy  = iface->super.config.seg_size - am_min_hdr;
    iface_attr->cap.am.max_hdr    = am_max_hdr - am_min_hdr;
    iface_attr->cap.am.max_iov    = am_max_iov;

    /* Error Handling */
    iface_attr->cap.flags        |= UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;

    if (iface_attr->cap.am.max_short) {
        iface_attr->cap.flags |= UCT_IFACE_FLAG_AM_SHORT;
    }

    if (iface_attr->cap.put.max_short) {
        iface_attr->cap.flags |= UCT_IFACE_FLAG_PUT_SHORT;
    }

    return UCS_OK;
}

void uct_rc_iface_add_qp(uct_rc_iface_t *iface, uct_rc_ep_t *ep,
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

void uct_rc_iface_remove_qp(uct_rc_iface_t *iface, unsigned qp_num)
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

    status = uct_rc_iface_fence_relaxed_order(tl_iface);
    if (status != UCS_OK) {
        return status;
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

    desc->lkey        = uct_ib_memh_get_lkey(memh);
    desc->super.flags = 0;
}

ucs_status_t uct_rc_init_fc_thresh(uct_rc_iface_config_t *config,
                                   uct_rc_iface_t *iface)
{
    /* Check FC parameters correctness */
    if ((config->soft_thresh <= config->super.fc.hard_thresh) ||
        (config->soft_thresh >= 1)) {
        ucs_error("The factor for soft FC threshold should be bigger"
                  " than FC_HARD_THRESH value and less than 1 (s=%f, h=%f)",
                  config->soft_thresh, config->super.fc.hard_thresh);
        return UCS_ERR_INVALID_PARAM;
    }

    if (config->super.fc.enable) {
        iface->config.fc_soft_thresh = ucs_max((int)(iface->config.fc_wnd_size *
                                               config->soft_thresh), 1);
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
    uct_rc_pending_req_t *fc_req;
    uct_rc_ep_t  *ep  = uct_rc_iface_lookup_ep(iface, qp_num);
    uint8_t fc_hdr    = uct_rc_fc_get_fc_hdr(hdr->am_id);

    ucs_assert(iface->config.fc_enabled);

    if (ep == NULL) {
        /* We get fc for ep which is being removed so should ignore it */
        goto out;
    }

    if (fc_hdr & UCT_RC_EP_FLAG_FC_GRANT) {
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

    if (fc_hdr & UCT_RC_EP_FLAG_FC_SOFT_REQ) {
        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_RX_SOFT_REQ, 1);

        /* Got soft credit request. Mark ep that it needs to grant
         * credits to the peer in outgoing AM (if any). */
        ep->flags |= UCT_RC_EP_FLAG_FC_GRANT;

    } else if (fc_hdr & UCT_RC_EP_FLAG_FC_HARD_REQ) {
        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_RX_HARD_REQ, 1);
        fc_req = ucs_mpool_get(&iface->tx.pending_mp);
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
            /* force add request to group & schedule group to eliminate
             * FC deadlock */
            uct_pending_req_arb_group_push_head(&iface->tx.arbiter,
                                                &ep->arb_group, &fc_req->super);
            ucs_arbiter_group_schedule(&iface->tx.arbiter, &ep->arb_group);
        } else {
            ucs_assertv_always(status == UCS_OK, "Failed to send FC grant msg: %s",
                               ucs_status_string(status));
        }
    }

out:
    return uct_iface_invoke_am(&iface->super.super,
                               (hdr->am_id & ~UCT_RC_EP_FC_MASK),
                               hdr + 1, length, flags);
}

static ucs_status_t uct_rc_iface_tx_ops_init(uct_rc_iface_t *iface)
{
    const unsigned count = iface->config.tx_ops_count;
    uct_rc_iface_send_op_t *op;
    ucs_status_t status;

    iface->tx.ops_buffer = ucs_calloc(count, sizeof(*iface->tx.ops_buffer),
                                      "rc_tx_ops");
    if (iface->tx.ops_buffer == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    iface->tx.free_ops = &iface->tx.ops_buffer[0];
    for (op = iface->tx.ops_buffer; op < iface->tx.ops_buffer + count; ++op) {
        op->handler = uct_rc_ep_send_op_completion_handler;
        op->flags   = UCT_RC_IFACE_SEND_OP_FLAG_IFACE;
        op->iface   = iface;
        op->next    = (op == (iface->tx.ops_buffer + count - 1)) ? NULL : (op + 1);
    }

    /* Create memory pool for flush completions. Can't just alloc a certain
     * size buffer, because number of simultaneous flushes is not limited by
     * CQ or QP resources. */
    status = ucs_mpool_init(&iface->tx.send_op_mp, 0, sizeof(*op), 0,
                            UCS_SYS_CACHE_LINE_SIZE, 256,
                            UINT_MAX, &uct_rc_send_op_mpool_ops,
                            "send-ops-mpool");

    return status;
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

    ucs_mpool_cleanup(&iface->tx.send_op_mp, 1);
}

unsigned uct_rc_iface_do_progress(uct_iface_h tl_iface)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);
    return iface->progress(iface);
}

ucs_status_t uct_rc_iface_init_rx(uct_rc_iface_t *iface,
                                  const uct_rc_iface_common_config_t *config,
                                  struct ibv_srq **srq_p)
{
    struct ibv_srq_init_attr srq_init_attr;
    struct ibv_pd *pd = uct_ib_iface_md(&iface->super)->pd;
    struct ibv_srq *srq;

    srq_init_attr.attr.max_sge   = 1;
    srq_init_attr.attr.max_wr    = config->super.rx.queue_len;
    srq_init_attr.attr.srq_limit = 0;
    srq_init_attr.srq_context    = iface;
    srq                          = ibv_create_srq(pd, &srq_init_attr);
    if (srq == NULL) {
        ucs_error("ibv_create_srq() failed: %m");
        return UCS_ERR_IO_ERROR;
    }
    iface->rx.srq.quota          = srq_init_attr.attr.max_wr;
    *srq_p                       = srq;

    return UCS_OK;
}

static int uct_rc_iface_config_limit_value(const char *name,
                                           int provided, int limit)
{
    if (provided > limit) {
         ucs_warn("using maximal value for %s (%d) instead of %d",
                  name, limit, provided);
         return limit;
     } else {
         return provided;
     }
}

UCS_CLASS_INIT_FUNC(uct_rc_iface_t, uct_rc_iface_ops_t *ops, uct_md_h md,
                    uct_worker_h worker, const uct_iface_params_t *params,
                    const uct_rc_iface_common_config_t *config,
                    uct_ib_iface_init_attr_t *init_attr)
{
    uct_ib_device_t *dev = &ucs_derived_of(md, uct_ib_md_t)->dev;
    uint32_t max_ib_msg_size;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_ib_iface_t, &ops->super, md, worker, params,
                              &config->super, init_attr);

    self->tx.cq_available       = init_attr->cq_len[UCT_IB_DIR_TX] - 1;
    self->rx.srq.available      = 0;
    self->rx.srq.quota          = 0;
    self->config.tx_qp_len      = config->super.tx.queue_len;
    self->config.tx_min_sge     = config->super.tx.min_sge;
    self->config.tx_min_inline  = config->super.tx.min_inline;
    self->config.tx_poll_always = config->tx.poll_always;
    self->config.tx_ops_count   = init_attr->cq_len[UCT_IB_DIR_TX];
    self->config.min_rnr_timer  = uct_ib_to_rnr_fabric_time(config->tx.rnr_timeout);
    self->config.timeout        = uct_ib_to_qp_fabric_time(config->tx.timeout);
    self->config.rnr_retry      = uct_rc_iface_config_limit_value(
                                                     "RNR_RETRY_COUNT",
                                                     config->tx.rnr_retry_count,
                                                     UCT_RC_QP_MAX_RETRY_COUNT);
    self->config.retry_cnt      = uct_rc_iface_config_limit_value(
                                                     "RETRY_COUNT",
                                                     config->tx.retry_count,
                                                     UCT_RC_QP_MAX_RETRY_COUNT);
    self->config.max_rd_atomic  = config->max_rd_atomic;
    self->config.ooo_rw         = config->ooo_rw;
#if UCS_ENABLE_ASSERT
    self->config.tx_cq_len      = init_attr->cq_len[UCT_IB_DIR_TX];
    self->tx.in_pending         = 0;
#endif
    max_ib_msg_size             = uct_ib_iface_port_attr(&self->super)->max_msg_sz;

    if (config->tx.max_get_zcopy == UCS_MEMUNITS_AUTO) {
        self->config.max_get_zcopy = max_ib_msg_size;
    } else if (config->tx.max_get_zcopy <= max_ib_msg_size) {
        self->config.max_get_zcopy = config->tx.max_get_zcopy;
    } else {
        ucs_warn("rc_iface on %s:%d: reduced max_get_zcopy to %u",
                 uct_ib_device_name(dev), self->super.config.port_num,
                 max_ib_msg_size);
        self->config.max_get_zcopy = max_ib_msg_size;
    }

    if ((config->tx.max_get_bytes == UCS_MEMUNITS_INF) ||
        (config->tx.max_get_bytes == UCS_MEMUNITS_AUTO)) {
        self->tx.reads_available = SSIZE_MAX;
    } else {
        self->tx.reads_available = config->tx.max_get_bytes;
    }

    self->tx.reads_completed = 0;

    uct_ib_fence_info_init(&self->tx.fi);
    memset(self->eps, 0, sizeof(self->eps));
    ucs_arbiter_init(&self->tx.arbiter);
    ucs_list_head_init(&self->ep_list);
    ucs_list_head_init(&self->ep_gc_list);

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

    /* Set atomic handlers according to atomic reply endianness */
    self->config.atomic64_handler = dev->atomic_arg_sizes_be & sizeof(uint64_t) ?
                                    uct_rc_ep_atomic_handler_64_be1 :
                                    uct_rc_ep_atomic_handler_64_be0;
    self->config.atomic32_ext_handler = dev->ext_atomic_arg_sizes_be & sizeof(uint32_t) ?
                                        uct_rc_ep_atomic_handler_32_be1 :
                                        uct_rc_ep_atomic_handler_32_be0;
    self->config.atomic64_ext_handler = dev->ext_atomic_arg_sizes_be & sizeof(uint64_t) ?
                                        uct_rc_ep_atomic_handler_64_be1 :
                                        uct_rc_ep_atomic_handler_64_be0;

    status = UCS_STATS_NODE_ALLOC(&self->stats, &uct_rc_iface_stats_class,
                                  self->super.super.stats);
    if (status != UCS_OK) {
        goto err_cleanup_tx_ops;
    }

    /* Initialize RX resources (SRQ) */
    status = ops->init_rx(self, config);
    if (status != UCS_OK) {
        goto err_destroy_stats;
    }

    /* Create mempool for pending requests */
    ucs_assert(init_attr->fc_req_size >= sizeof(uct_rc_pending_req_t));
    status = ucs_mpool_init(&self->tx.pending_mp, 0, init_attr->fc_req_size,
                            0, 1, 128, UINT_MAX, &uct_rc_pending_mpool_ops,
                            "pending-ops");
    if (status != UCS_OK) {
        goto err_cleanup_rx;
    }

    self->config.fc_enabled = config->fc.enable;
    if (self->config.fc_enabled) {
        /* Assume that number of recv buffers is the same on all peers.
         * Then FC window size is the same for all endpoints as well.
         * TODO: Make wnd size to be a property of the particular interface.
         * We could distribute it via rc address then.*/
        self->config.fc_wnd_size    = ucs_min(config->fc.wnd_size,
                                              config->super.rx.queue_len);
        self->config.fc_hard_thresh = ucs_max((int)(self->config.fc_wnd_size *
                                              config->fc.hard_thresh), 1);
    } else {
        self->config.fc_wnd_size    = INT16_MAX;
        self->config.fc_hard_thresh = 0;
    }

    return UCS_OK;

err_cleanup_rx:
    ops->cleanup_rx(self);
err_destroy_stats:
    UCS_STATS_NODE_FREE(self->stats);
err_cleanup_tx_ops:
    uct_rc_iface_tx_ops_cleanup(self);
err_destroy_tx_mp:
    ucs_mpool_cleanup(&self->tx.mp, 1);
err_destroy_rx_mp:
    ucs_mpool_cleanup(&self->rx.mp, 1);
err:
    return status;
}

void uct_rc_iface_cleanup_eps(uct_rc_iface_t *iface)
{
    uct_rc_iface_ops_t *ops = ucs_derived_of(iface->super.ops, uct_rc_iface_ops_t);
    uct_rc_ep_cleanup_ctx_t *cleanup_ctx, *tmp;

    ucs_list_for_each_safe(cleanup_ctx, tmp, &iface->ep_gc_list, list) {
        ops->cleanup_qp(&cleanup_ctx->super);
    }

    ucs_assert(ucs_list_is_empty(&iface->ep_gc_list));
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_iface_t)
{
    uct_rc_iface_ops_t *ops = ucs_derived_of(self->super.ops, uct_rc_iface_ops_t);
    unsigned i;

    /* Release table. TODO release on-demand when removing ep. */
    for (i = 0; i < UCT_RC_QP_TABLE_SIZE; ++i) {
        ucs_free(self->eps[i]);
    }

    if (!ucs_list_is_empty(&self->ep_list)) {
        ucs_warn("some eps were not destroyed");
    }

    ucs_arbiter_cleanup(&self->tx.arbiter);

    UCS_STATS_NODE_FREE(self->stats);

    ops->cleanup_rx(self);
    uct_rc_iface_tx_ops_cleanup(self);
    ucs_mpool_cleanup(&self->tx.mp, 1);
    ucs_mpool_cleanup(&self->rx.mp, 0); /* Cannot flush SRQ */
    ucs_mpool_cleanup(&self->tx.pending_mp, 1);
}

UCS_CLASS_DEFINE(uct_rc_iface_t, uct_ib_iface_t);

void uct_rc_iface_fill_attr(uct_rc_iface_t *iface, uct_ib_qp_attr_t *attr,
                            unsigned max_send_wr, struct ibv_srq *srq)
{
    attr->srq                        = srq;
    attr->cap.max_send_wr            = max_send_wr;
    attr->cap.max_recv_wr            = 0;
    attr->cap.max_send_sge           = iface->config.tx_min_sge;
    attr->cap.max_recv_sge           = 1;
    attr->cap.max_inline_data        = iface->config.tx_min_inline;
    attr->qp_type                    = iface->super.config.qp_type;
    attr->sq_sig_all                 = !iface->config.tx_moderation;
    attr->max_inl_cqe[UCT_IB_DIR_RX] = iface->super.config.max_inl_cqe[UCT_IB_DIR_RX];
    attr->max_inl_cqe[UCT_IB_DIR_TX] = iface->super.config.max_inl_cqe[UCT_IB_DIR_TX];
}

ucs_status_t uct_rc_iface_qp_create(uct_rc_iface_t *iface, struct ibv_qp **qp_p,
                                    uct_ib_qp_attr_t *attr, unsigned max_send_wr,
                                    struct ibv_srq *srq)
{
    uct_rc_iface_fill_attr(iface, attr, max_send_wr, srq);
    uct_ib_iface_fill_attr(&iface->super, attr);

    return uct_ib_iface_create_qp(&iface->super, attr, qp_p);
}

ucs_status_t uct_rc_iface_qp_init(uct_rc_iface_t *iface, struct ibv_qp *qp)
{
    struct ibv_qp_attr qp_attr;
    int ret;

    memset(&qp_attr, 0, sizeof(qp_attr));

    qp_attr.qp_state              = IBV_QPS_INIT;
    qp_attr.pkey_index            = iface->super.pkey_index;
    qp_attr.port_num              = iface->super.config.port_num;
    qp_attr.qp_access_flags       = IBV_ACCESS_LOCAL_WRITE  |
                                    IBV_ACCESS_REMOTE_WRITE |
                                    IBV_ACCESS_REMOTE_READ  |
                                    IBV_ACCESS_REMOTE_ATOMIC;
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
                                     struct ibv_ah_attr *ah_attr,
                                     enum ibv_mtu path_mtu)
{
#if HAVE_DECL_IBV_EXP_QP_OOO_RW_DATA_PLACEMENT
    struct ibv_exp_qp_attr qp_attr;
    uct_ib_device_t *dev;
#else
    struct ibv_qp_attr qp_attr;
#endif
    long qp_attr_mask;
    int ret;

    ucs_assert(path_mtu != 0);

    memset(&qp_attr, 0, sizeof(qp_attr));

    qp_attr.qp_state              = IBV_QPS_RTR;
    qp_attr.dest_qp_num           = dest_qp_num;
    qp_attr.rq_psn                = 0;
    qp_attr.path_mtu              = path_mtu;
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
    dev = uct_ib_iface_device(&iface->super);
    if (iface->config.ooo_rw && UCX_IB_DEV_IS_OOO_SUPPORTED(dev, rc)) {
        ucs_debug("enabling out-of-order on RC QP %x dev %s",
                  qp->qp_num, uct_ib_device_name(dev));
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

ucs_status_t uct_rc_iface_common_event_arm(uct_iface_h tl_iface,
                                           unsigned events, int force_rx_all)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);
    int arm_rx_solicited, arm_rx_all;
    ucs_status_t status;

    status = uct_ib_iface_pre_arm(&iface->super);
    if (status != UCS_OK) {
        return status;
    }

    if (events & UCT_EVENT_SEND_COMP) {
        status = iface->super.ops->arm_cq(&iface->super, UCT_IB_DIR_TX, 0);
        if (status != UCS_OK) {
            return status;
        }
    }

    arm_rx_solicited = 0;
    arm_rx_all       = 0;
    if (events & UCT_EVENT_RECV) {
        arm_rx_solicited = 1; /* to wake up on active messages */
    }
    if (((events & UCT_EVENT_SEND_COMP) && iface->config.fc_enabled) ||
        force_rx_all) {
        arm_rx_all       = 1; /* to wake up on FC grants (or if forced) */
    }

    if (arm_rx_solicited || arm_rx_all) {
        status = iface->super.ops->arm_cq(&iface->super, UCT_IB_DIR_RX,
                                          arm_rx_solicited && !arm_rx_all);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;

}

ucs_status_t uct_rc_iface_event_arm(uct_iface_h tl_iface, unsigned events)
{
    return uct_rc_iface_common_event_arm(tl_iface, events, 0);
}

ucs_status_t uct_rc_iface_fence(uct_iface_h tl_iface, unsigned flags)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);

    if (iface->config.fence_mode != UCT_RC_FENCE_MODE_NONE) {
        iface->tx.fi.fence_beat++;
    }

    UCT_TL_IFACE_STAT_FENCE(&iface->super.super);
    return UCS_OK;
}
