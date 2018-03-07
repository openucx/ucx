/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "dc_iface.h"
#include "dc_ep.h"

#include <uct/ib/base/ib_device.h>
#include <ucs/async/async.h>


const static char *uct_dc_tx_policy_names[] = {
    [UCT_DC_TX_POLICY_DCS]           = "dcs",
    [UCT_DC_TX_POLICY_DCS_QUOTA]     = "dcs_quota",
    [UCT_DC_TX_POLICY_LAST]          = NULL
};

ucs_config_field_t uct_dc_iface_config_table[] = {
    {"RC_", "IB_TX_QUEUE_LEN=128;FC_ENABLE=n;", NULL,
     ucs_offsetof(uct_dc_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_rc_iface_config_table)},

    {"", "", NULL,
     ucs_offsetof(uct_dc_iface_config_t, ud_common),
     UCS_CONFIG_TYPE_TABLE(uct_ud_iface_common_config_table)},

    {"NUM_DCI", "8",
     "Number of DC initiator QPs (DCI) used by the interface "
     "(up to " UCS_PP_QUOTE(UCT_DC_IFACE_MAX_DCIS) ").",
     ucs_offsetof(uct_dc_iface_config_t, ndci), UCS_CONFIG_TYPE_UINT},

    {"TX_POLICY", "dcs_quota",
     "Specifies how DC initiator (DCI) is selected by the endpoint. The policies are:\n"
     "\n"
     "dcs        The endpoint either uses already assigned DCI or one is allocated\n"
     "           in a LIFO order, and released once it has no outstanding operations.\n"
     "\n"
     "dcs_quota  Same as \"dcs\" but in addition the DCI is scheduled for release\n"
     "           if it has sent more than quota, and there are endpoints waiting for a DCI.\n"
     "           The dci is released once it completes all outstanding operations.\n"
     "           This policy ensures that there will be no starvation among endpoints.",
     ucs_offsetof(uct_dc_iface_config_t, tx_policy),
     UCS_CONFIG_TYPE_ENUM(uct_dc_tx_policy_names)},

    {"QUOTA", "32",
     "When \"dcs_quota\" policy is selected, how much to send from a DCI when\n"
     "there are other endpoints waiting for it.",
     ucs_offsetof(uct_dc_iface_config_t, quota), UCS_CONFIG_TYPE_UINT},

    {NULL}
};

ucs_status_t uct_dc_iface_create_dct(uct_dc_iface_t *iface)
{
    struct ibv_exp_dct_init_attr init_attr;

    memset(&init_attr, 0, sizeof(init_attr));

    init_attr.pd               = uct_ib_iface_md(&iface->super.super)->pd;
    init_attr.cq               = iface->super.super.recv_cq;
    init_attr.srq              = iface->super.rx.srq.srq;
    init_attr.dc_key           = UCT_IB_KEY;
    init_attr.port             = iface->super.super.config.port_num;
    init_attr.mtu              = iface->super.config.path_mtu;
    init_attr.access_flags     = IBV_EXP_ACCESS_REMOTE_WRITE |
                                 IBV_EXP_ACCESS_REMOTE_READ |
                                 IBV_EXP_ACCESS_REMOTE_ATOMIC;
    init_attr.min_rnr_timer    = iface->super.config.min_rnr_timer;
    init_attr.hop_limit        = 1;
    init_attr.inline_size      = iface->super.config.rx_inline;

#if HAVE_DECL_IBV_EXP_DCT_OOO_RW_DATA_PLACEMENT
    if (iface->super.config.ooo_rw &&
        UCX_IB_DEV_IS_OOO_SUPPORTED(&uct_ib_iface_device(&iface->super.super)->dev_attr,
                                    dc)) {
        ucs_debug("creating DC target with out-of-order support dev %s",
                   uct_ib_device_name(uct_ib_iface_device(&iface->super.super)));
        init_attr.create_flags |= IBV_EXP_DCT_OOO_RW_DATA_PLACEMENT;
    }
#endif

    iface->rx.dct = ibv_exp_create_dct(uct_ib_iface_device(&iface->super.super)->ibv_context,
                                       &init_attr);
    if (iface->rx.dct == NULL) {
        ucs_error("Failed to created DC target %m");
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

/* take dc qp to rts state */
static ucs_status_t uct_dc_iface_dci_connect(uct_dc_iface_t *iface,
                                             uct_rc_txqp_t *dci)
{
    struct ibv_exp_qp_attr attr;
    long attr_mask;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state        = IBV_QPS_INIT;
    attr.pkey_index      = 0;
    attr.qp_access_flags = 0;
    attr.port_num        = iface->super.super.config.port_num;
    attr.dct_key         = UCT_IB_KEY;

    if (ibv_exp_modify_qp(dci->qp, &attr,
                         IBV_EXP_QP_STATE        |
                         IBV_EXP_QP_PKEY_INDEX   |
                         IBV_EXP_QP_PORT         |
                         IBV_EXP_QP_DC_KEY
                         )) {
        ucs_error("error modifying QP to INIT : %m");
        return UCS_ERR_IO_ERROR;
    }

    /* Move QP to the RTR state */
    memset(&attr, 0, sizeof(attr));
    attr.qp_state                   = IBV_QPS_RTR;
    attr.path_mtu                   = iface->super.config.path_mtu;
    attr.min_rnr_timer              = 0;
    attr.max_dest_rd_atomic         = 1;
    if ((iface->super.super.addr_type == UCT_IB_ADDRESS_TYPE_ETH) ||
        (iface->super.super.addr_type == UCT_IB_ADDRESS_TYPE_GLOBAL))
    {
        attr.ah_attr.is_global      = 1;
    }
    attr.ah_attr.sl                 = iface->super.super.config.sl;
    attr_mask                       = IBV_EXP_QP_STATE     |
                                      IBV_EXP_QP_PATH_MTU  |
                                      IBV_EXP_QP_AV;

#if HAVE_DECL_IBV_EXP_QP_OOO_RW_DATA_PLACEMENT
    if (iface->super.config.ooo_rw &&
        UCX_IB_DEV_IS_OOO_SUPPORTED(&uct_ib_iface_device(&iface->super.super)->dev_attr,
                                    dc)) {
        ucs_debug("enabling out-of-order on DCI QP 0x%x dev %s", dci->qp->qp_num,
                   uct_ib_device_name(uct_ib_iface_device(&iface->super.super)));
        attr_mask |= IBV_EXP_QP_OOO_RW_DATA_PLACEMENT;
    }
#endif

    if (ibv_exp_modify_qp(dci->qp, &attr, attr_mask)) {
        ucs_error("error modifying QP to RTR: %m");
        return UCS_ERR_IO_ERROR;
    }

    /* Move QP to the RTS state */
    memset(&attr, 0, sizeof(attr));
    attr.qp_state       = IBV_QPS_RTS;
    attr.timeout        = iface->super.config.timeout;
    attr.rnr_retry      = iface->super.config.rnr_retry;
    attr.retry_cnt      = iface->super.config.retry_cnt;
    attr.max_rd_atomic  = iface->super.config.max_rd_atomic;
    attr_mask           = IBV_EXP_QP_STATE      |
                          IBV_EXP_QP_TIMEOUT    |
                          IBV_EXP_QP_RETRY_CNT  |
                          IBV_EXP_QP_RNR_RETRY  |
                          IBV_EXP_QP_MAX_QP_RD_ATOMIC;

    if (ibv_exp_modify_qp(dci->qp, &attr, attr_mask)) {
        ucs_error("error modifying QP to RTS: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

static void uct_dc_iface_dcis_destroy(uct_dc_iface_t *iface, int max)
{
    int i;
    for (i = 0; i < max; i++) {
        uct_rc_txqp_cleanup(&iface->tx.dcis[i].txqp);
    }
}

static ucs_status_t uct_dc_iface_create_dcis(uct_dc_iface_t *iface,
                                             uct_dc_iface_config_t *config)
{
    struct ibv_qp_cap cap;
    ucs_status_t status;
    int i;

    ucs_debug("creating %d dci(s)", iface->tx.ndci);

    iface->tx.stack_top = 0;
    for (i = 0; i < iface->tx.ndci; i++) {
        status = uct_rc_txqp_init(&iface->tx.dcis[i].txqp, &iface->super,
                                  IBV_EXP_QPT_DC_INI, &cap
                                  UCS_STATS_ARG(iface->super.stats));
        if (status != UCS_OK) {
            goto err;
        }

        status = uct_dc_iface_dci_connect(iface, &iface->tx.dcis[i].txqp);
        if (status != UCS_OK) {
            uct_rc_txqp_cleanup(&iface->tx.dcis[i].txqp);
            goto err;
        }

        iface->tx.dcis_stack[i] = i;
        iface->tx.dcis[i].ep    = NULL;
#if ENABLE_ASSERT
        iface->tx.dcis[i].flags = 0;
#endif
    }
    uct_ib_iface_set_max_iov(&iface->super.super, cap.max_send_sge);
    return UCS_OK;

err:
    uct_dc_iface_dcis_destroy(iface, i);
    return status;
}

void uct_dc_iface_set_quota(uct_dc_iface_t *iface, uct_dc_iface_config_t *config)
{
    iface->tx.available_quota = iface->super.config.tx_qp_len -
                                ucs_min(iface->super.config.tx_qp_len, config->quota);
}

static void uct_dc_iface_init_version(uct_dc_iface_t *iface, uct_md_h md)
{
    uct_ib_device_t *dev;
    unsigned         ver;

    dev = &ucs_derived_of(md, uct_ib_md_t)->dev;
    ver = uct_ib_device_spec(dev)->flags & UCT_IB_DEVICE_FLAG_DC;
    ucs_assert(ver != UCT_IB_DEVICE_FLAG_DC);

    iface->version_flag = 0;

    if (ver & UCT_IB_DEVICE_FLAG_DC_V2) {
        iface->version_flag = UCT_DC_IFACE_ADDR_DC_V2;
    }

    if (ver & UCT_IB_DEVICE_FLAG_DC_V1) {
        iface->version_flag = UCT_DC_IFACE_ADDR_DC_V1;
    }
}

UCS_CLASS_INIT_FUNC(uct_dc_iface_t, uct_dc_iface_ops_t *ops, uct_md_h md,
                    uct_worker_h worker, const uct_iface_params_t *params,
                    unsigned rx_priv_len, uct_dc_iface_config_t *config,
                    int tm_cap_bit)
{
    ucs_status_t status;
    ucs_trace_func("");

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_iface_t, &ops->super, md, worker, params,
                              &config->super, rx_priv_len,
                              sizeof(uct_dc_fc_request_t), tm_cap_bit);
    if (config->ndci < 1) {
        ucs_error("dc interface must have at least 1 dci (requested: %d)",
                  config->ndci);
        return UCS_ERR_INVALID_PARAM;
    }

    if (config->ndci > UCT_DC_IFACE_MAX_DCIS) {
        ucs_error("dc interface can have at most %d dcis (requested: %d)",
                  UCT_DC_IFACE_MAX_DCIS, config->ndci);
        return UCS_ERR_INVALID_PARAM;
    }

    uct_dc_iface_init_version(self, md);

    self->tx.ndci                    = config->ndci;
    self->tx.policy                  = config->tx_policy;
    self->tx.available_quota         = 0; /* overridden by mlx5/verbs */
    self->super.config.tx_moderation = 0; /* disable tx moderation for dcs */
    ucs_list_head_init(&self->tx.gc_list);

    /* create DC target */
    if (!UCT_RC_IFACE_TM_ENABLED(&self->super)) {
        status = uct_dc_iface_create_dct(self);
        if (status != UCS_OK) {
            goto err;
        }
    }

    /* create DC initiators */
    status = uct_dc_iface_create_dcis(self, config);
    if (status != UCS_OK) {
        goto err_destroy_dct;
    }

    ucs_debug("dc iface %p: using '%s' policy with %d dcis, dct 0x%x", self,
              uct_dc_tx_policy_names[self->tx.policy], self->tx.ndci,
              UCT_RC_IFACE_TM_ENABLED(&self->super) ?
              0 : self->rx.dct->dct_num);

    /* Create fake endpoint which will be used for sending FC grants */
    uct_dc_iface_init_fc_ep(self);

    ucs_arbiter_init(&self->tx.dci_arbiter);
    return UCS_OK;

err_destroy_dct:
    if (!UCT_RC_IFACE_TM_ENABLED(&self->super)) {
        ibv_exp_destroy_dct(self->rx.dct);
    }
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_iface_t)
{
    uct_dc_ep_t *ep, *tmp;

    ucs_trace_func("");
    if (self->rx.dct != NULL) {
        ibv_exp_destroy_dct(self->rx.dct);
    }
    ucs_list_for_each_safe(ep, tmp, &self->tx.gc_list, list) {
        uct_dc_ep_release(ep);
    }
    uct_dc_iface_dcis_destroy(self, self->tx.ndci);
    ucs_arbiter_cleanup(&self->tx.dci_arbiter);
    uct_dc_iface_cleanup_fc_ep(self);
}

UCS_CLASS_DEFINE(uct_dc_iface_t, uct_rc_iface_t);

ucs_status_t uct_dc_iface_query(uct_dc_iface_t *iface,
                                uct_iface_attr_t *iface_attr,
                                size_t put_max_short, size_t max_inline,
                                size_t am_max_hdr, size_t am_max_iov)
{
    ucs_status_t status;

    status = uct_rc_iface_query(&iface->super, iface_attr, put_max_short,
                                max_inline, am_max_hdr, am_max_iov);
    if (status != UCS_OK) {
        return status;
    }

    /* fixup flags and address lengths */
    iface_attr->cap.flags &= ~UCT_IFACE_FLAG_CONNECT_TO_EP;
    iface_attr->cap.flags |= UCT_IFACE_FLAG_CONNECT_TO_IFACE;
    iface_attr->ep_addr_len       = 0;
    iface_attr->max_conn_priv     = 0;
    iface_attr->iface_addr_len    = sizeof(uct_dc_iface_addr_t);
    iface_attr->latency.overhead += 60e-9; /* connect packet + cqe */

    return UCS_OK;
}

int uct_dc_iface_is_reachable(const uct_iface_h tl_iface,
                              const uct_device_addr_t *dev_addr,
                              const uct_iface_addr_t *iface_addr)
{
    uct_dc_iface_t UCS_V_UNUSED *iface = ucs_derived_of(tl_iface,
                                                        uct_dc_iface_t);
    uct_dc_iface_addr_t *addr = (uct_dc_iface_addr_t *)iface_addr;

    ucs_assert_always(iface_addr != NULL);

    return ((addr->flags & UCT_DC_IFACE_ADDR_DC_VERS) == iface->version_flag) &&
           (UCT_DC_IFACE_ADDR_TM_ENABLED(addr) ==
            UCT_RC_IFACE_TM_ENABLED(&iface->super)) &&
           uct_ib_iface_is_reachable(tl_iface, dev_addr, iface_addr);
}

ucs_status_t
uct_dc_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    uct_dc_iface_t      *iface = ucs_derived_of(tl_iface, uct_dc_iface_t);
    uct_dc_iface_addr_t *addr  = (uct_dc_iface_addr_t *)iface_addr;

    uct_ib_pack_uint24(addr->qp_num, iface->rx.dct->dct_num);
    addr->atomic_mr_id = uct_ib_iface_get_atomic_mr_id(&iface->super.super);
    addr->flags        = iface->version_flag;
    if (UCT_RC_IFACE_TM_ENABLED(&iface->super)) {
        addr->flags   |= UCT_DC_IFACE_ADDR_HW_TM;
    }

    return UCS_OK;
}

ucs_status_t uct_dc_device_query_tl_resources(uct_ib_device_t *dev,
                                              const char *tl_name, unsigned flags,
                                              uct_tl_resource_desc_t **resources_p,
                                              unsigned *num_resources_p)
{
    return uct_ib_device_query_tl_resources(dev, tl_name,
                                            flags | UCT_IB_DEVICE_FLAG_DC,
                                            resources_p, num_resources_p);
}

static inline ucs_status_t uct_dc_iface_flush_dcis(uct_dc_iface_t *iface)
{
    int i;
    int is_flush_done = 1;

    for (i = 0; i < iface->tx.ndci; i++) {
        if ((iface->tx.dcis[i].ep != NULL) &&
            uct_dc_ep_fc_wait_for_grant(iface->tx.dcis[i].ep)) {
            return UCS_ERR_NO_RESOURCE;
        }
        if (uct_dc_iface_flush_dci(iface, i) != UCS_OK) {
            is_flush_done = 0;
        }
    }
    return is_flush_done ? UCS_OK : UCS_INPROGRESS;
}

ucs_status_t uct_dc_iface_flush(uct_iface_h tl_iface, unsigned flags, uct_completion_t *comp)
{
    uct_dc_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_iface_t);
    ucs_status_t status;

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }
    status = uct_dc_iface_flush_dcis(iface);
    if (status == UCS_OK) {
        UCT_TL_IFACE_STAT_FLUSH(&iface->super.super.super);
    }
    else if (status == UCS_INPROGRESS) {
        UCT_TL_IFACE_STAT_FLUSH_WAIT(&iface->super.super.super);
    }
    return status;
}

ucs_status_t uct_dc_iface_init_fc_ep(uct_dc_iface_t *iface)
{
    ucs_status_t status;
    uct_dc_ep_t *ep;

    ep = ucs_malloc(sizeof(uct_dc_ep_t), "fc_ep");
    if (ep == NULL) {
        ucs_error("Failed to allocate FC ep");
        status =  UCS_ERR_NO_MEMORY;
        goto err;
    }
    /* We do not have any peer address at this point, so init basic subclasses
     * only (for statistics, iface, etc) */
    status = UCS_CLASS_INIT(uct_base_ep_t, (void*)(&ep->super),
                            &iface->super.super.super);
    if (status != UCS_OK) {
        ucs_error("Failed to initialize fake FC ep, status: %s",
                  ucs_status_string(status));
        goto err_free;
    }

    status = uct_dc_ep_basic_init(iface, ep);
    if (status != UCS_OK) {
        ucs_error("FC ep init failed %s", ucs_status_string(status));
        goto err_cleanup;
    }

    iface->tx.fc_ep = ep;
    return UCS_OK;

err_cleanup:
    UCS_CLASS_CLEANUP(uct_base_ep_t, &ep->super);
err_free:
    ucs_free(ep);
err:
    return status;
}

void uct_dc_iface_cleanup_fc_ep(uct_dc_iface_t *iface)
{
    uct_dc_ep_pending_purge(&iface->tx.fc_ep->super.super, NULL, NULL);
    ucs_arbiter_group_cleanup(&iface->tx.fc_ep->arb_group);
    uct_rc_fc_cleanup(&iface->tx.fc_ep->fc);
    UCS_CLASS_CLEANUP(uct_base_ep_t, iface->tx.fc_ep);
    ucs_free(iface->tx.fc_ep);
}

ucs_status_t uct_dc_iface_fc_grant(uct_pending_req_t *self)
{
    ucs_status_t status;
    uct_rc_fc_request_t *freq = ucs_derived_of(self, uct_rc_fc_request_t);
    uct_dc_ep_t *ep           = ucs_derived_of(freq->ep, uct_dc_ep_t);
    uct_rc_iface_t *iface     = ucs_derived_of(ep->super.super.iface,
                                               uct_rc_iface_t);

    ucs_assert_always(iface->config.fc_enabled);

    status = uct_rc_fc_ctrl(&ep->super.super, UCT_RC_EP_FC_PURE_GRANT, freq);
    if (status == UCS_OK) {
        ucs_mpool_put(freq);
        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_TX_PURE_GRANT, 1);
    }
    return status;
}

ucs_status_t uct_dc_iface_fc_handler(uct_rc_iface_t *rc_iface, unsigned qp_num,
                                     uct_rc_hdr_t *hdr, unsigned length,
                                     uint32_t imm_data, uint16_t lid, unsigned flags)
{
    uct_dc_ep_t *ep;
    ucs_status_t status;
    int16_t      cur_wnd;
    uct_dc_fc_request_t *dc_req;
    uint8_t fc_hdr        = uct_rc_fc_get_fc_hdr(hdr->am_id);
    uct_dc_iface_t *iface = ucs_derived_of(rc_iface, uct_dc_iface_t);

    ucs_assert(rc_iface->config.fc_enabled);

    if (fc_hdr == UCT_RC_EP_FC_FLAG_HARD_REQ) {
        ep = iface->tx.fc_ep;
        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_RX_HARD_REQ, 1);

        dc_req = ucs_mpool_get(&iface->super.tx.fc_mp);
        if (ucs_unlikely(dc_req == NULL)) {
            ucs_error("Failed to allocate FC request");
            return UCS_ERR_NO_MEMORY;
        }
        dc_req->super.super.func = uct_dc_iface_fc_grant;
        dc_req->super.ep         = &ep->super.super;
        dc_req->dct_num          = imm_data;
        dc_req->lid              = lid;
        dc_req->sender           = *((uct_dc_fc_sender_data_t*)(hdr + 1));

        status = uct_dc_iface_fc_grant(&dc_req->super.super);
        if (status == UCS_ERR_NO_RESOURCE){
            status = uct_ep_pending_add(&ep->super.super, &dc_req->super.super);
        }
        ucs_assertv_always(status == UCS_OK, "Failed to send FC grant msg: %s",
                           ucs_status_string(status));
    } else if (fc_hdr == UCT_RC_EP_FC_PURE_GRANT) {
        ep = *((uct_dc_ep_t**)(hdr + 1));

        if (!(ep->flags & UCT_DC_EP_FLAG_VALID)) {
            uct_dc_ep_release(ep);
            return UCS_OK;
        }

        cur_wnd = ep->fc.fc_wnd;

        /* Peer granted resources, so update wnd */
        ep->fc.fc_wnd = rc_iface->config.fc_wnd_size;

        /* Clear the flag for flush to complete  */
        ep->fc.flags &= ~UCT_DC_EP_FC_FLAG_WAIT_FOR_GRANT;

        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_RX_PURE_GRANT, 1);
        UCS_STATS_SET_COUNTER(ep->fc.stats, UCT_RC_FC_STAT_FC_WND, ep->fc.fc_wnd);

        /* To preserve ordering we have to dispatch all pending
         * operations if current fc_wnd is <= 0 */
        if (cur_wnd <= 0) {
            if (ep->dci == UCT_DC_EP_NO_DCI) {
                ucs_arbiter_group_schedule(uct_dc_iface_dci_waitq(iface),
                                           &ep->arb_group);
            } else {
                /* Need to schedule fake ep in TX arbiter, because it
                 * might have been descheduled due to lack of FC window. */
                ucs_arbiter_group_schedule(uct_dc_iface_tx_waitq(iface),
                                           &ep->arb_group);
            }

            uct_dc_iface_progress_pending(iface);
        }
    }

    return UCS_OK;
}

ucs_status_t uct_dc_handle_failure(uct_ib_iface_t *ib_iface, uint32_t qp_num,
                                   ucs_status_t status)
{
    uct_dc_iface_t     *iface  = ucs_derived_of(ib_iface, uct_dc_iface_t);
    uint8_t            dci     = uct_dc_iface_dci_find(iface, qp_num);
    uct_rc_txqp_t      *txqp   = &iface->tx.dcis[dci].txqp;
    uct_dc_ep_t        *ep     = iface->tx.dcis[dci].ep;
    uct_dc_iface_ops_t *dc_ops = ucs_derived_of(iface->super.super.ops,
                                                uct_dc_iface_ops_t);
    ucs_status_t       ep_status;
    int16_t            outstanding;

    if (!ep) {
        return UCS_OK;
    }

    uct_rc_txqp_purge_outstanding(txqp, status, 0);

    /* poll_cqe for mlx5 returns NULL in case of failure and the cq_avaialble
       is not updated for the error cqe and all outstanding wqes*/
    outstanding = (int16_t)iface->super.config.tx_qp_len -
                  uct_rc_txqp_available(txqp);
    iface->super.tx.cq_available += outstanding;
    uct_rc_txqp_available_set(txqp, (int16_t)iface->super.config.tx_qp_len);

    /* since we removed all outstanding ops on the dci, it should be released */
    ucs_assert(ep->dci != UCT_DC_EP_NO_DCI);
    uct_dc_iface_dci_put(iface, dci);
    ucs_assert_always(ep->dci == UCT_DC_EP_NO_DCI);

    ep_status = iface->super.super.ops->set_ep_failed(ib_iface,
                                                      &ep->super.super, status);

    status = dc_ops->reset_dci(iface, dci);
    if (status != UCS_OK) {
        ucs_fatal("iface %p failed to reset dci[%d] qpn 0x%x: %s",
                   iface, dci, txqp->qp->qp_num, ucs_status_string(status));
    }

    status = uct_dc_iface_dci_connect(iface, txqp);
    if (status != UCS_OK) {
        ucs_fatal("iface %p failed to connect dci[%d] qpn 0x%x: %s",
                  iface, dci, txqp->qp->qp_num, ucs_status_string(status));
    }

    return ep_status;
}

#if IBV_EXP_HW_TM_DC
void uct_dc_iface_fill_xrq_init_attrs(uct_rc_iface_t *rc_iface,
                                      struct ibv_exp_create_srq_attr *srq_attr,
                                      struct ibv_exp_srq_dc_offload_params *dc_op)
{
    dc_op->timeout    = rc_iface->config.timeout;
    dc_op->path_mtu   = rc_iface->config.path_mtu;
    dc_op->pkey_index = 0;
    dc_op->sl         = rc_iface->super.config.sl;
    dc_op->dct_key    = UCT_IB_KEY;

    srq_attr->comp_mask         = IBV_EXP_CREATE_SRQ_DC_OFFLOAD_PARAMS;
    srq_attr->dc_offload_params = dc_op;
}
#endif
