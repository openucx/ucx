/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) 2007-2009 Cisco Systems, Inc.  All rights reserved.
* Copyright (c) 2009      IBM Corporation.  All rights reserved.
*
* See file LICENSE for terms.
*/

#include "cm.h"

#include <ucs/arch/atomic.h>
#include <ucs/async/async.h>
#include <uct/base/uct_log.h>


typedef struct uct_cm_iov {
    uct_pack_callback_t pack;
    const void          *arg;
    size_t              length;
} uct_cm_iov_t;


static UCS_CLASS_INIT_FUNC(uct_cm_ep_t, uct_iface_t *tl_iface,
                           const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)

{
    uct_cm_iface_t *iface = ucs_derived_of(tl_iface, uct_cm_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    uct_ib_address_unpack((const uct_ib_address_t*)dev_addr, &self->dlid,
                          &self->is_global, &self->dgid);
    self->dest_service_id = *(const uint32_t*)iface_addr;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cm_ep_t)
{
    ucs_trace_func("");
}

UCS_CLASS_DEFINE(uct_cm_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_cm_ep_t, uct_ep_t, uct_iface_h,
                          const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_cm_ep_t, uct_ep_t);


static ucs_status_t uct_cm_ep_fill_path_rec(uct_cm_ep_t *ep,
                                            struct ibv_sa_path_rec *path)
{
    uct_cm_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_cm_iface_t);

    path->sgid                      = iface->super.gid;
    path->dlid                      = htons(ep->dlid);
    path->slid                      = htons(uct_ib_iface_port_attr(&iface->super)->lid);
    if (ep->is_global) {
        path->dgid                  = ep->dgid;
        path->hop_limit             = 64;
    } else {
        memset(&path->dgid, 0, sizeof(path->dgid));
        path->hop_limit             = 0;
    }
    path->raw_traffic               = 0; /* IB traffic */
    path->flow_label                = 0;
    path->traffic_class             = iface->super.config.traffic_class;
    path->reversible                = htonl(1); /* IBCM currently only supports reversible paths */
    path->numb_path                 = 0;
    path->pkey                      = ntohs(iface->super.pkey_value);
    path->sl                        = iface->super.config.sl;
    path->mtu_selector              = 2; /* EQ */
    path->mtu                       = uct_ib_iface_port_attr(&iface->super)->active_mtu;
    path->rate_selector             = 2; /* EQ */
    path->rate                      = IBV_RATE_MAX;
    path->packet_life_time_selector = 2; /* EQ */
    path->packet_life_time          = 0;
    path->preference                = 0; /* Use first path */
    return UCS_OK;
}

static void uct_cm_dump_path(struct ibv_sa_path_rec *path)
{
    char sgid_buf[256];
    char dgid_buf[256];

    inet_ntop(AF_INET6, &path->dgid, dgid_buf, sizeof(dgid_buf));
    inet_ntop(AF_INET6, &path->sgid, sgid_buf, sizeof(sgid_buf));

    ucs_trace_data("slid %d sgid %s dlid %d dgid %s",
                   ntohs(path->slid), sgid_buf, ntohs(path->dlid), dgid_buf);
    ucs_trace_data("traffic %d flow_label %d hop %d class %d revers. 0x%x "
                   "numb %d pkey 0x%x sl %d",
                   path->raw_traffic, path->flow_label, path->hop_limit,
                   path->traffic_class, path->reversible, path->numb_path,
                   path->pkey, path->sl);
    ucs_trace_data("mtu %d(%d) rate %d(%d) lifetime %d(%d) pref %d",
                   path->mtu, path->mtu_selector, path->rate, path->rate_selector,
                   path->packet_life_time, path->packet_life_time_selector,
                   path->preference);
}

ssize_t uct_cm_ep_am_bcopy(uct_ep_h tl_ep, uint8_t am_id, uct_pack_callback_t pack_cb,
                           void *arg, unsigned flags)
{
    uct_cm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cm_iface_t);
    uct_cm_ep_t *ep = ucs_derived_of(tl_ep, uct_cm_ep_t);
    struct ib_cm_sidr_req_param req;
    struct ibv_sa_path_rec path;
    uct_cm_iface_op_t *op;
    ucs_status_t status;
    uct_cm_hdr_t *hdr;
    size_t payload_len;
    size_t total_len;
    int ret;

    UCT_CHECK_AM_ID(am_id);

    uct_cm_enter(iface);

    if (!uct_cm_iface_has_tx_resources(iface)) {
        status = UCS_ERR_NO_RESOURCE;
        goto err;
    }

    /* Allocate temporary contiguous buffer */
    hdr = ucs_malloc(IB_CM_SIDR_REQ_PRIVATE_DATA_SIZE, "cm_send_buf");
    if (hdr == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    payload_len = pack_cb(hdr + 1, arg);
    hdr->am_id  = am_id;
    hdr->length = payload_len;
    total_len   = sizeof(*hdr) + payload_len;

    status = uct_cm_ep_fill_path_rec(ep, &path);
    if (status != UCS_OK) {
        goto err_free_hdr;
    }

    /* Fill SIDR request */
    memset(&req, 0, sizeof req);
    req.path             = &path;
    req.service_id       = ep->dest_service_id;
    req.timeout_ms       = iface->config.timeout_ms;
    req.private_data     = hdr;
    req.private_data_len = total_len;
    req.max_cm_retries   = iface->config.retry_count;

    op = ucs_malloc(sizeof *op, "cm_op");
    if (op == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_hdr;
    }

    op->is_id = 1;

    /* Create temporary ID for this message. Will be released when getting REP. */
    ret = ib_cm_create_id(iface->cmdev, &op->id, NULL);
    if (ret) {
        ucs_error("ib_cm_create_id() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_free_op;
    }

    uct_cm_dump_path(&path);

    ret = ib_cm_send_sidr_req(op->id, &req);
    if (ret) {
        ucs_error("ib_cm_send_sidr_req() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_destroy_id;
    }

    ucs_queue_push(&iface->outstanding_q, &op->queue);
    ++iface->num_outstanding;
    ucs_trace("outs=%d", iface->num_outstanding);
    UCT_TL_EP_STAT_OP(&ep->super, AM, BCOPY, payload_len);

    uct_cm_iface_trace_data(iface, UCT_AM_TRACE_TYPE_SEND, hdr,
                            "TX: SIDR_REQ [id %p{%u} dlid %d svc 0x%"PRIx64"]",
                            op->id, op->id->handle, ntohs(path.dlid),
                            (uint64_t)req.service_id);
    uct_cm_leave(iface);
    ucs_free(hdr);
    /* coverity[missing_unlock] */
    return payload_len;

err_destroy_id:
    ib_cm_destroy_id(op->id);
err_free_op:
    ucs_free(op);
err_free_hdr:
    ucs_free(hdr);
err:
    uct_cm_leave(iface);
    return status;
}

ucs_status_t uct_cm_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *req)
{
    uct_cm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cm_iface_t);
    uct_cm_ep_t *ep = ucs_derived_of(tl_ep, uct_cm_ep_t);
    ucs_status_t status;

    uct_cm_enter(iface);
    if (iface->num_outstanding < iface->config.max_outstanding) {
        status = UCS_ERR_BUSY;
    } else {
        ucs_derived_of(uct_pending_req_priv(req), uct_cm_pending_req_priv_t)->ep = ep;
        uct_pending_req_push(&iface->notify_q, req);
        status = UCS_OK;
    }
    uct_cm_leave(iface);
    return status;
}

void uct_cm_ep_pending_purge(uct_ep_h tl_ep, uct_pending_purge_callback_t cb,
                             void *arg)
{
    uct_cm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cm_iface_t);
    uct_cm_ep_t *ep = ucs_derived_of(tl_ep, uct_cm_ep_t);
    uct_cm_pending_req_priv_t *priv;

    uct_cm_enter(iface);
    uct_pending_queue_purge(priv, &iface->notify_q, priv->ep == ep, cb, arg);
    uct_cm_leave(iface);
}

ucs_status_t uct_cm_ep_flush(uct_ep_h tl_ep, unsigned flags,
                             uct_completion_t *comp)
{
    uct_cm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cm_iface_t);
    ucs_status_t status;

    uct_cm_enter(iface);
    if (!uct_cm_iface_has_tx_resources(iface)) {
        status = UCS_ERR_NO_RESOURCE;
    } else {
        status = uct_cm_iface_flush_do(iface, comp);
        if (status == UCS_OK) {
            UCT_TL_EP_STAT_FLUSH(ucs_derived_of(tl_ep, uct_base_ep_t));
        } else if (status == UCS_INPROGRESS) {
            UCT_TL_EP_STAT_FLUSH_WAIT(ucs_derived_of(tl_ep, uct_base_ep_t));
        }
    }
    uct_cm_leave(iface);

    return status;
}
