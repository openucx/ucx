/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ud_iface.h"
#include "ud_ep.h"

#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/type/class.h>

static ucs_status_t uct_ud_iface_tx_mpool_create(uct_ib_iface_t *iface,
                                                 uct_ib_iface_config_t *config, 
                                                 const char *name, ucs_mpool_h *mp_p);

static ucs_status_t uct_ud_iface_create_qp(uct_ud_iface_t *self, uct_ud_iface_config_t *config)
{
    /* TODO: exp attrs autoconf */
    struct ibv_exp_qp_init_attr qp_init_attr;
    struct ibv_qp_attr qp_attr;
    uct_ib_device_t *dev;
    int ret;

    dev = uct_ib_iface_device(&self->super);

    /* Create QP */
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.qp_context          = NULL;
    qp_init_attr.send_cq             = self->super.send_cq;
    qp_init_attr.recv_cq             = self->super.recv_cq;
    qp_init_attr.srq                 = NULL; /* TODO */
    qp_init_attr.qp_type             = IBV_QPT_UD;
    qp_init_attr.sq_sig_all          = 0;
    
    /* TODO: cap setting */
    qp_init_attr.cap.max_send_wr     = config->super.tx.queue_len;
    qp_init_attr.cap.max_recv_wr     = config->super.rx.queue_len;
    qp_init_attr.cap.max_send_sge    = 2;
    qp_init_attr.cap.max_recv_sge    = 1;
    qp_init_attr.cap.max_inline_data = ucs_max(config->super.tx.min_inline,
                                               UCT_UD_MIN_INLINE);

#if HAVE_VERBS_EXP_H
    qp_init_attr.pd                  = dev->pd;
    qp_init_attr.comp_mask           = IBV_QP_INIT_ATTR_PD;
    /* TODO: inline rcv */
#if 0
    if (mxm_ud_ep_opts(ep)->ud.ib.rx.max_inline > 0) {
        qp_init_attr.comp_mask      |= IBV_EXP_QP_INIT_ATTR_INL_RECV;
        qp_init_attr.max_inl_recv    = mxm_ud_ep_opts(ep)->ud.ib.rx.max_inline;
    }
#endif
    self->qp = ibv_exp_create_qp(dev->ibv_context, &qp_init_attr);
#else
    self->qp = ibv_exp_create_qp(dev->pd, &qp_init_attr);
#endif
    if (self->qp == NULL) {
        ucs_error("Failed to create qp: %m [inline: %u rsge: %u ssge: %u rwr: %u swr: %u]",
                qp_init_attr.cap.max_inline_data, qp_init_attr.cap.max_recv_sge,
                qp_init_attr.cap.max_send_sge, qp_init_attr.cap.max_recv_wr,
                qp_init_attr.cap.max_send_wr);
        goto err;
    }

    memset(&qp_attr, 0, sizeof(qp_attr));
    /* Modify QP to INIT state */
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = self->super.port_num;
    qp_attr.qkey = UCT_UD_QKEY;
    ret = ibv_modify_qp(self->qp, &qp_attr,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY);
    if (ret) {
        ucs_error("Failed to modify UD QP to INIT: %m");
        goto err_destroy_qp;
    }

    /* Modify to RTR */
    qp_attr.qp_state = IBV_QPS_RTR;
    ret = ibv_modify_qp(self->qp, &qp_attr, IBV_QP_STATE);
    if (ret) {
        ucs_error("Failed to modify UD QP to RTR: %m");
        goto err_destroy_qp;
    }

    /* Modify to RTS */
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.sq_psn = 0;
    ret = ibv_modify_qp(self->qp, &qp_attr, IBV_QP_STATE | IBV_QP_SQ_PSN);
    if (ret) {
        ucs_error("Failed to modify UD QP to RTS: %m");
        goto err_destroy_qp;
    }

    ucs_debug("iface=%p: created qp 0x%x max_send_wr %u max_recv_wr %u max_inline %u",
            self, self->qp->qp_num,
            qp_init_attr.cap.max_send_wr,
            qp_init_attr.cap.max_recv_wr,
            qp_init_attr.cap.max_inline_data);

    return UCS_OK;
err_destroy_qp:
    ibv_destroy_qp(self->qp);
err:
    return UCS_ERR_INVALID_PARAM;
}

UCS_CLASS_INIT_FUNC(uct_ud_iface_t, uct_iface_ops_t *ops, uct_worker_h worker,
                    const char *dev_name, unsigned rx_headroom, unsigned rx_priv_len,
                    uct_ud_iface_config_t *config)
{
    ucs_status_t status;

    ucs_trace_func("%s: iface=%p ops=%p worker=%p rx_headroom=%u rx_priv_len=%u",
                   dev_name, self, ops, worker, rx_headroom, rx_priv_len);

    UCS_CLASS_CALL_SUPER_INIT(uct_ib_iface_t, ops, worker, dev_name, rx_headroom,
                              rx_priv_len + sizeof(uct_ud_recv_skb_t) - sizeof(uct_ib_iface_recv_desc_t), 
                              UCT_IB_GRH_LEN + sizeof(uct_ud_neth_t),
                              config->super.tx.queue_len, &config->super);
 
    self->tx.unsignaled          = 0;
    self->tx.available           = config->super.tx.queue_len;

    self->rx.available           = config->super.rx.queue_len;
    self->config.tx_qp_len       = config->super.tx.queue_len;
    self->config.rx_max_batch    = ucs_min(config->super.rx.max_batch, config->super.rx.queue_len / 4);

    if (uct_ud_iface_create_qp(self, config) != UCS_OK) {
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_ptr_array_init(&self->eps, 0, "ud_eps");

    /* TODO: correct hdr + payload offset  */
    status = uct_ib_iface_recv_mpool_create(&self->super, &config->super,
                                            "ud_recv_skb", &self->rx.mp); 
    if (status != UCS_OK) {
        goto err_qp;
    }

    status = uct_ud_iface_tx_mpool_create(&self->super, &config->super,
                                          "ud_tx_skb", &self->tx.mp); 
    if (status != UCS_OK) {
        goto err_mpool;
    }

    ucs_queue_head_init(&self->tx.pending_ops);
    return UCS_OK;

err_mpool:
    ucs_mpool_destroy(self->rx.mp);
err_qp:
    ibv_destroy_qp(self->qp);
    ucs_ptr_array_cleanup(&self->eps);
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ud_iface_t)
{
    ucs_trace_func("");

    /* TODO: proper flush and connection termination */
    ucs_mpool_destroy_unchecked(self->tx.mp);
    /* TODO: qp to error state and cleanup all wqes */
    ucs_mpool_destroy_unchecked(self->rx.mp);
    ibv_destroy_qp(self->qp);
    ucs_ptr_array_cleanup(&self->eps);
}

UCS_CLASS_DEFINE(uct_ud_iface_t, uct_ib_iface_t);

ucs_config_field_t uct_ud_iface_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_ud_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_ib_iface_config_table)},
    {NULL}
};


void uct_ud_iface_query(uct_ud_iface_t *iface, uct_iface_attr_t *iface_attr)
{
    struct ibv_qp_init_attr qp_init_attr;
    struct ibv_qp_attr qp_attr;
    int mtu = 4096; /* TODO: mtu from port header */
    int ret;

    /* Get QP properties */
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    memset(&qp_attr, 0, sizeof(qp_attr));
    ret = ibv_query_qp(iface->qp, &qp_attr, IBV_QP_CAP, &qp_init_attr);
    if (ret != 0) {
        ucs_fatal("ibv_query_qp() failed: %m");
    }

    memset(iface_attr, 0, sizeof(*iface_attr));
    iface_attr->cap.flags           = UCT_IFACE_FLAG_AM_SHORT; /* | UCT_IFACE_FLAG_PUT_SHORT; */ 

    ucs_assert(qp_attr.cap.max_inline_data > UCT_UD_MIN_INLINE);
    iface_attr->cap.am.max_short      = qp_attr.cap.max_inline_data - sizeof(uct_ud_neth_t);
    iface_attr->cap.am.max_bcopy      = mtu - sizeof(uct_ud_neth_t);
    iface_attr->cap.am.max_zcopy      = 0;

    iface_attr->cap.put.max_short     = qp_attr.cap.max_inline_data - sizeof(uct_ud_neth_t) - sizeof(uct_ud_put_hdr_t);
    iface_attr->cap.am.max_bcopy      = mtu - sizeof(uct_ud_neth_t);
    iface_attr->cap.am.max_zcopy      = 0;

    iface_attr->iface_addr_len        = sizeof(uct_ud_iface_addr_t);
    iface_attr->ep_addr_len           = sizeof(uct_ud_ep_addr_t);
    iface_attr->completion_priv_len   = 0;

}

ucs_status_t uct_ud_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);
    uct_ud_iface_addr_t *addr = (uct_ud_iface_addr_t *)iface_addr;
    uct_ib_device_t *dev;
    uint32_t lid;

    addr->qp_num = iface->qp->qp_num;
    dev = uct_ib_iface_device(&iface->super);
    lid = dev->port_attr[iface->super.port_num-dev->first_port].lid;
    addr->lid = lid; 
    ucs_debug("qpnum=%d lid=%d", addr->qp_num, addr->lid);

    return UCS_OK;
}

ucs_status_t uct_ud_iface_flush(uct_iface_h tl_iface)
{
#if 0
    /* TODO: flush code */
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);

    ucs_trace_func("");
    if (iface->tx.available < iface->config.tx_qp_len) {
        /* TODO: flush eps */
        return UCS_ERR_WOULD_BLOCK;
    }
#endif
    usleep(100);
    return UCS_OK;
}

void uct_ud_iface_add_ep(uct_ud_iface_t *iface, uct_ud_ep_t *ep)
{
    uint32_t prev_gen;
    ep->ep_id = ucs_ptr_array_insert(&iface->eps, ep, &prev_gen);
}

void uct_ud_iface_remove_ep(uct_ud_iface_t *iface, uct_ud_ep_t *ep)
{
    ucs_ptr_array_remove(&iface->eps, ep->ep_id, 0);
}

static void uct_ud_iface_send_skb_init(uct_iface_h tl_iface, void *obj, uct_mem_h memh)
{       
    uct_ud_send_skb_t *skb = obj;
    struct ibv_mr *mr = memh;
    skb->lkey = mr->lkey;
}


static ucs_status_t uct_ud_iface_tx_mpool_create(uct_ib_iface_t *iface,
                                                 uct_ib_iface_config_t *config, 
                                                 const char *name, ucs_mpool_h *mp_p)
{
    unsigned grow;
    int mtu;

    if (config->tx.queue_len < 1024) {
        grow = 1024;
    }

    mtu = 4096; /* TODO: calculate mtu */

    return uct_iface_mpool_create(&iface->super.super,
                                  sizeof(uct_ud_send_skb_t) + mtu,
                                  0,
                                  UCS_SYS_CACHE_LINE_SIZE,
                                  &config->tx.mp,
                                  grow,
                                  uct_ud_iface_send_skb_init,
                                  name,
                                  mp_p);

}

