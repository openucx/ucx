/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "dc_iface.h"

static ucs_status_t uct_dc_iface_tgt_create(uct_dc_iface_t *iface)
{
    struct ibv_exp_dct_init_attr init_attr;

    memset(&init_attr, 0, sizeof(init_attr));

    init_attr.pd               = uct_ib_iface_pd(&iface->super.super)->pd;
    init_attr.cq               = iface->super.super.recv_cq;
    init_attr.srq              = iface->super.rx.srq;
    init_attr.dc_key           = UCT_IB_DC_KEY;
    init_attr.port             = iface->super.super.port_num;
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
        ucs_error("Failed to created DC target");
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
    attr.port_num        = iface->super.super.port_num;
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
    attr.ah_attr.sl                 = iface->super.super.sl;

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

static ucs_status_t uct_dc_iface_dcis_create(uct_dc_iface_t *iface)
{
    ucs_status_t status;
    int i;
    struct ibv_qp_cap cap;

    iface->tx.ndci = 8; /* TODO: make configurable */

    iface->tx.dcis = ucs_malloc(iface->tx.ndci * sizeof(uct_rc_txqp_t), "dc");
    if (iface->tx.dcis == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    for (i = 0; i < iface->tx.ndci; i++) {
        status = uct_rc_txqp_init(&iface->tx.dcis[i], &iface->super, 
                                  IBV_EXP_QPT_DC_INI, &cap);
        if (status != UCS_OK) {
            goto create_err;
        }

        status = uct_dc_iface_dci_connect(iface, &iface->tx.dcis[i]);
        if (status != UCS_OK) {
            goto create_err;
        }
    }
    iface->config.max_inline = cap.max_inline_data;
    return UCS_OK;

create_err:
    for (;i >= 0; i--) {
        if (iface->tx.dcis[i].qp) { 
            ibv_destroy_qp(iface->tx.dcis[i].qp);
        }
    }
    return status;
}

UCS_CLASS_INIT_FUNC(uct_dc_iface_t, uct_rc_iface_ops_t *ops, uct_pd_h pd,
                    uct_worker_h worker, const char *dev_name, unsigned rx_headroom,
                    unsigned rx_priv_len, uct_dc_iface_config_t *config)
{
    ucs_status_t status;
    ucs_trace_func("");

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_iface_t, ops, pd, worker, dev_name, rx_headroom,
                              rx_priv_len, &config->super);

    /* create DC target */
    status = uct_dc_iface_tgt_create(self);
    if (status != UCS_OK) {
        return status;
    }

    /* create DC initiators */
    status = uct_dc_iface_dcis_create(self);
    if (status != UCS_OK) {
        ibv_exp_destroy_dct(self->rx.dct);
        return status;
    }
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_iface_t)
{
   int i;

    ucs_trace_func("");
    ibv_exp_destroy_dct(self->rx.dct);

    for (i = 0; i < self->tx.ndci; i++) {
        uct_rc_txqp_cleanup(&self->tx.dcis[i]);
    }
    ucs_free(self->tx.dcis);
}


UCS_CLASS_DEFINE(uct_dc_iface_t, uct_rc_iface_t);

ucs_config_field_t uct_dc_iface_config_table[] = {
    {"IB_", "", NULL,
        ucs_offsetof(uct_dc_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_rc_iface_config_table)},
    {NULL}
};

void uct_dc_iface_query(uct_dc_iface_t *iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_iface_query(&iface->super, iface_attr);

    /* fixup flags and address lengths */
    iface_attr->cap.flags           = UCT_IFACE_FLAG_AM_SHORT|UCT_IFACE_FLAG_AM_CB_SYNC; /* todo remove */
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

    return UCS_OK;
}

ucs_status_t uct_dc_device_query_tl_resources(uct_ib_device_t *dev,
                                              const char *tl_name, unsigned flags,
                                              uct_tl_resource_desc_t **resources_p,
                                              unsigned *num_resources_p)
{
    return uct_ib_device_query_tl_resources(dev, tl_name, flags|UCT_IB_DEVICE_FLAG_DC, 
                                            resources_p, num_resources_p);
}

