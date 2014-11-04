/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_ep.h"
#include "rc_iface.h"

#include <uct/ib/base/ib_verbs.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>


ucs_status_t uct_rc_ep_init(uct_rc_ep_t *ep)
{
    uct_rc_iface_t *iface = ucs_derived_of(ep->super.iface, uct_rc_iface_t);
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super);
    struct ibv_qp_init_attr qp_init_attr;
    ucs_status_t status;

    /* Create QP */
    qp_init_attr.qp_context          = NULL;
    qp_init_attr.send_cq             = iface->super.send_cq;
    qp_init_attr.recv_cq             = iface->super.recv_cq;
    qp_init_attr.srq                 = NULL; /* TODO */
    qp_init_attr.cap.max_send_wr     = 1024;
    qp_init_attr.cap.max_recv_wr     = 1024;
    qp_init_attr.cap.max_send_sge    = 2;
    qp_init_attr.cap.max_recv_sge    = 1;
    qp_init_attr.cap.max_inline_data = 0;
    qp_init_attr.qp_type             = IBV_QPT_RC;
    qp_init_attr.sq_sig_all          = 0;
    qp_init_attr.xrc_domain          = NULL;
    ep->qp = ibv_create_qp(dev->pd, &qp_init_attr);
    if (ep->qp == NULL) {
        ucs_error("failed to create qp: %m");
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    return UCS_OK;

err:
    return status;
}

void uct_rc_ep_cleanup(uct_rc_ep_t *ep)
{
    int ret;

    ret = ibv_destroy_qp(ep->qp);
    if (ret != 0) {
        ucs_warn("ibv_destroy_qp() returned %d: %m", ret);
    }
}

ucs_status_t uct_rc_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *ep_addr)
{
    uct_rc_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_ep_t);

    ((uct_rc_ep_addr_t*)ep_addr)->qp_num = ep->qp->qp_num;
    return UCS_OK;
}

ucs_status_t uct_rc_ep_connect_to_ep(uct_ep_h tl_ep, uct_iface_addr_t *tl_iface_addr,
                                     uct_ep_addr_t *tl_ep_addr)
{
    uct_rc_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_ep_t);
    uct_rc_iface_t *iface = ucs_derived_of(ep->super.iface, uct_rc_iface_t);
    uct_ib_iface_addr_t *iface_addr = ucs_derived_of(tl_iface_addr, uct_ib_iface_addr_t);
    uct_rc_ep_addr_t *ep_addr = ucs_derived_of(tl_ep_addr, uct_rc_ep_addr_t);
    struct ibv_qp_attr qp_attr;
    int ret;

    memset(&qp_attr, 0, sizeof(qp_attr));

    qp_attr.qp_state              = IBV_QPS_INIT;
    qp_attr.pkey_index            = 0;
    qp_attr.port_num              = iface->super.port_num;
    qp_attr.qp_access_flags       = IBV_ACCESS_LOCAL_WRITE|
                                    IBV_ACCESS_REMOTE_WRITE|
                                    IBV_ACCESS_REMOTE_READ|
                                    IBV_ACCESS_REMOTE_ATOMIC;
    ret = ibv_modify_qp(ep->qp, &qp_attr,
                        IBV_QP_STATE      |
                        IBV_QP_PKEY_INDEX |
                        IBV_QP_PORT       |
                        IBV_QP_ACCESS_FLAGS);
    if (ret) {
         ucs_error("error modifying QP to INIT: %m");
         return UCS_ERR_IO_ERROR;
    }

    qp_attr.qp_state              = IBV_QPS_RTR;
    qp_attr.ah_attr.dlid          = iface_addr->lid; /* TODO LMC */
    qp_attr.ah_attr.sl            = 9; /* TODO SL */
    qp_attr.ah_attr.src_path_bits = 0; /* TODO LMC */
    qp_attr.ah_attr.static_rate   = 0;
    qp_attr.ah_attr.is_global     = 0; /* TODO RoCE */
    qp_attr.ah_attr.port_num      = iface->super.port_num;
    qp_attr.dest_qp_num           = ep_addr->qp_num;
    qp_attr.rq_psn                = 0;
    qp_attr.path_mtu              = IBV_MTU_2048; /* TODO select by device type */
    qp_attr.max_dest_rd_atomic    = 4; /* TODO consider this in sender */
    qp_attr.min_rnr_timer         = 14; /* TODO config */
    ret = ibv_modify_qp(ep->qp, &qp_attr,
                        IBV_QP_STATE              |
                        IBV_QP_AV                 |
                        IBV_QP_PATH_MTU           |
                        IBV_QP_DEST_QPN           |
                        IBV_QP_RQ_PSN             |
                        IBV_QP_MAX_DEST_RD_ATOMIC |
                        IBV_QP_MIN_RNR_TIMER);
    if (ret) {
        ucs_error("error modifying QP to RTR: %m");
        return UCS_ERR_IO_ERROR;
    }

    qp_attr.qp_state              = IBV_QPS_RTS;
    qp_attr.sq_psn                = 0;
    qp_attr.timeout               = 14; /* TODO config */
    qp_attr.rnr_retry             = 7; /* TODO config */
    qp_attr.retry_cnt             = 7; /* TODO config */
    qp_attr.max_rd_atomic         = 4; /* TODO consider this in sender */
    ret = ibv_modify_qp(ep->qp, &qp_attr,
                        IBV_QP_STATE              |
                        IBV_QP_TIMEOUT            |
                        IBV_QP_RETRY_CNT          |
                        IBV_QP_RNR_RETRY          |
                        IBV_QP_SQ_PSN             |
                        IBV_QP_MAX_QP_RD_ATOMIC);
    if (ret) {
        ucs_error("error modifying QP to RTS: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

