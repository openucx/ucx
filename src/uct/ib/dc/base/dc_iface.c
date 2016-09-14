/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "dc_iface.h"

const static char *uct_dc_tx_policy_names[] = {
    [UCT_DC_TX_POLICY_DCS]           = "dcs",
    [UCT_DC_TX_POLICY_DCS_QUOTA]     = "dcs_quota",
    [UCT_DC_TX_POLICY_LAST]          = NULL
};

static ucs_status_t uct_dc_iface_tgt_create(uct_dc_iface_t *iface)
{
    struct ibv_exp_dct_init_attr init_attr;

    memset(&init_attr, 0, sizeof(init_attr));

    init_attr.pd               = uct_ib_iface_md(&iface->super.super)->pd;
    init_attr.cq               = iface->super.super.recv_cq;
    init_attr.srq              = iface->super.rx.srq;
    init_attr.dc_key           = UCT_IB_DC_KEY;
    init_attr.port             = iface->super.super.config.port_num;
    init_attr.mtu              = iface->super.config.path_mtu;
    init_attr.access_flags     = IBV_EXP_ACCESS_REMOTE_WRITE |
                                 IBV_EXP_ACCESS_REMOTE_READ | 
                                 IBV_EXP_ACCESS_REMOTE_ATOMIC;
    init_attr.min_rnr_timer    = iface->super.config.min_rnr_timer;
    init_attr.hop_limit        = 1;
    init_attr.inline_size      = iface->super.config.rx_inline;

    iface->rx.dct = ibv_exp_create_dct(uct_ib_iface_device(&iface->super.super)->ibv_context, 
                                       &init_attr);
    if (iface->rx.dct == NULL) { 
        ucs_error("Failed to created DC target %m");
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

/* take dc qp to rts state */
static ucs_status_t uct_dc_iface_dci_connect(uct_dc_iface_t *iface, uct_rc_txqp_t *dci) 
{
    struct ibv_exp_qp_attr attr;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state        = IBV_QPS_INIT;
    attr.pkey_index      = 0;
    attr.qp_access_flags = 0;
    attr.port_num        = iface->super.super.config.port_num;
    attr.dct_key         = UCT_IB_DC_KEY;

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
    attr.ah_attr.sl                 = iface->super.super.config.sl;

    if (ibv_exp_modify_qp(dci->qp, &attr,
                         IBV_EXP_QP_STATE     |
                         IBV_EXP_QP_PATH_MTU  |
                         IBV_EXP_QP_AV)) {
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

    if (ibv_exp_modify_qp(dci->qp, &attr,
                         IBV_EXP_QP_STATE      |
                         IBV_EXP_QP_TIMEOUT    |
                         IBV_EXP_QP_RETRY_CNT  |
                         IBV_EXP_QP_RNR_RETRY  |
                         IBV_EXP_QP_MAX_QP_RD_ATOMIC)) {
        ucs_error("error modifying QP to RTS: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

static ucs_status_t uct_dc_iface_dcis_create(uct_dc_iface_t *iface, uct_dc_iface_config_t *config)
{
    ucs_status_t status;
    int i;
    struct ibv_qp_cap cap;

    ucs_debug("creating %d dci(s)", iface->tx.ndci);
    iface->tx.dcis = ucs_malloc(iface->tx.ndci * sizeof(uct_dc_dci_t), "dc");
    if (iface->tx.dcis == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    iface->tx.dcis_stack = ucs_malloc(iface->tx.ndci * sizeof(uint8_t), "dc");
    if (iface->tx.dcis_stack == NULL) {
        free(iface->tx.dcis);
        return UCS_ERR_NO_MEMORY;
    }

    iface->tx.stack_top = 0;

    for (i = 0; i < iface->tx.ndci; i++) {
        status = uct_rc_txqp_init(&iface->tx.dcis[i].txqp, &iface->super, 
                                  IBV_EXP_QPT_DC_INI, &cap 
                                  UCS_STATS_ARG(iface->super.stats));
        if (status != UCS_OK) {
            goto create_err;
        }

        status = uct_dc_iface_dci_connect(iface, &iface->tx.dcis[i].txqp);
        if (status != UCS_OK) {
            goto create_err;
        }

        iface->tx.dcis_stack[i] = i;
        iface->tx.dcis[i].ep    = NULL;
    }
    config->max_inline = cap.max_inline_data;
    return UCS_OK;

create_err:
    for (;i >= 0; i--) {
        if (iface->tx.dcis[i].txqp.qp) { 
            ibv_destroy_qp(iface->tx.dcis[i].txqp.qp);
        }
    }
    ucs_free(iface->tx.dcis);
    ucs_free(iface->tx.dcis_stack);
    return status;
}

UCS_CLASS_INIT_FUNC(uct_dc_iface_t, uct_rc_iface_ops_t *ops, uct_md_h md,
                    uct_worker_h worker, const char *dev_name, unsigned rx_headroom,
                    unsigned rx_priv_len, uct_dc_iface_config_t *config)
{
    ucs_status_t status;
    ucs_trace_func("");

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_iface_t, ops, md, worker, dev_name, rx_headroom,
                              rx_priv_len, &config->super.super);

    self->tx.ndci   = config->ndci;
    self->tx.policy = config->tx_policy;
    ucs_debug("using %s dci selection algorithm",
              uct_dc_tx_policy_names[self->tx.policy]);
    /* create DC target */
    status = uct_dc_iface_tgt_create(self);
    if (status != UCS_OK) {
        return status;
    }

    /* create DC initiators */
    status = uct_dc_iface_dcis_create(self, config);
    if (status != UCS_OK) {
        ibv_exp_destroy_dct(self->rx.dct);
        return status;
    }
    ucs_arbiter_init(&self->tx.dci_arbiter);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_iface_t)
{
   int i;

    ucs_trace_func("");
    ibv_exp_destroy_dct(self->rx.dct);

    for (i = 0; i < self->tx.ndci; i++) {
        uct_rc_txqp_cleanup(&self->tx.dcis[i].txqp);
    }
    ucs_free(self->tx.dcis);
    ucs_free(self->tx.dcis_stack);
    ucs_arbiter_cleanup(&self->tx.dci_arbiter);
}


UCS_CLASS_DEFINE(uct_dc_iface_t, uct_rc_iface_t);

ucs_config_field_t uct_dc_iface_config_table[] = {
    {"DC_", "", NULL,
        ucs_offsetof(uct_dc_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_rc_verbs_iface_config_table)},
    {"NUM_DCI", "8", 
     "number of QPs (dynamic connection initiators) allocated by the interface",
     ucs_offsetof(uct_dc_iface_config_t, ndci), UCS_CONFIG_TYPE_UINT},

    {"TX_POLICY", "dcs_quota", 
     "Specifies how dci is selected by the endpoint. The policies are:\n"
     "\n"
     "dcs        the endpoint either uses already assigned dci or a dci is allocated in the LIFO order.\n"
     "           The dci is released once it has no outstanding operations.\n"      
     "\n"
     "dcs_quota  same as dcs. In addition the dci is scheduled for release\n"
     "           if it can not transmit and there are endpoints waiting for the dci allocation.\n"
     "           The dci is released once it completes all outstanding operations.\n"  
     "           The policy ensures that there will be no starvation among endpoints.",
     ucs_offsetof(uct_dc_iface_config_t, tx_policy), UCS_CONFIG_TYPE_ENUM(uct_dc_tx_policy_names)},

    {NULL}
};

void uct_dc_iface_query(uct_dc_iface_t *iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_iface_query(&iface->super, iface_attr);

    /* fixup flags and address lengths */
    iface_attr->cap.flags &= ~UCT_IFACE_FLAG_CONNECT_TO_EP;
    iface_attr->cap.flags |= UCT_IFACE_FLAG_CONNECT_TO_IFACE;
    iface_attr->ep_addr_len    = 0;
    iface_attr->iface_addr_len = sizeof(uct_dc_iface_addr_t);

}

ucs_status_t
uct_dc_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    uct_dc_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_iface_t);
    uct_dc_iface_addr_t *addr = (uct_dc_iface_addr_t *)iface_addr;

    uct_ib_pack_uint24(addr->qp_num, iface->rx.dct->dct_num);
    addr->umr_id = uct_ib_iface_umr_id(&iface->super.super);
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

