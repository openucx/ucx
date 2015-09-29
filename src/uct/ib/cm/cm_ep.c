/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) 2007-2009 Cisco Systems, Inc.  All rights reserved.
* Copyright (c) 2009      IBM Corporation.  All rights reserved.
*
* See file LICENSE for terms.
*/

#include "cm.h"

#include <ucs/async/async.h>
#include <uct/tl/tl_log.h>
#include <infiniband/arch.h>


typedef struct uct_cm_iov {
    uct_pack_callback_t pack;
    const void          *arg;
    size_t              length;
} uct_cm_iov_t;


static UCS_CLASS_INIT_FUNC(uct_cm_ep_t, uct_iface_t *tl_iface, const struct sockaddr *addr)

{
    uct_cm_iface_t *iface = ucs_derived_of(tl_iface, uct_cm_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);
    self->dest_addr  = *(const uct_sockaddr_ib_t*)addr;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cm_ep_t)
{
    ucs_trace_func("");
}

UCS_CLASS_DEFINE(uct_cm_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_cm_ep_t, uct_ep_t, uct_iface_h, const struct sockaddr *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_cm_ep_t, uct_ep_t);


static ucs_status_t uct_cm_ep_fill_path_rec(uct_cm_ep_t *ep,
                                            struct ibv_sa_path_rec *path)
{
    uct_cm_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_cm_iface_t);

    path->dgid.global.subnet_prefix = ep->dest_addr.subnet_prefix;
    path->dgid.global.interface_id  = ep->dest_addr.guid;
    path->sgid                      = iface->super.gid;
    path->dlid                      = htons(ep->dest_addr.lid);
    path->slid                      = htons(uct_ib_iface_port_attr(&iface->super)->lid);
    path->raw_traffic               = 0; /* IB traffic */
    path->flow_label                = 0;
    path->hop_limit                 = 0;
    path->traffic_class             = 0;
    path->reversible                = htonl(1); /* IBCM currently only supports reversible paths */
    path->numb_path                 = 0;
    path->pkey                      = ntohs(iface->super.pkey_value);
    path->sl                        = iface->super.sl;
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

ssize_t uct_cm_ep_am_bcopy(uct_ep_h tl_ep, uint8_t am_id,
                           uct_pack_callback_t pack_cb, void *arg)
{
    uct_cm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cm_iface_t);
    uct_cm_ep_t *ep = ucs_derived_of(tl_ep, uct_cm_ep_t);
    struct ib_cm_sidr_req_param req;
    struct ibv_sa_path_rec path;
    struct ib_cm_id *id;
    ucs_status_t status;
    uct_cm_hdr_t *hdr;
    size_t payload_len;
    size_t total_len;
    int ret;

    UCT_CHECK_AM_ID(am_id);

    if (ucs_atomic_fadd32(&iface->outstanding, 1) >= iface->config.max_outstanding) {
        status = UCS_ERR_NO_RESOURCE;
        goto err_dec_outstanding;
    }

    /* Allocate temporary contiguous buffer */
    hdr = ucs_malloc(IB_CM_SIDR_REQ_PRIVATE_DATA_SIZE, "cm_send_buf");
    if (hdr == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_dec_outstanding;
    }

    payload_len = pack_cb(hdr + 1, arg);
    hdr->am_id  = am_id;
    hdr->length = payload_len;
    total_len   = sizeof(*hdr) + payload_len;

    status = uct_cm_ep_fill_path_rec(ep, &path);
    if (status != UCS_OK) {
        goto err_free;
    }

    /* Fill SIDR request */
    memset(&req, 0, sizeof req);
    req.path             = &path;
    req.service_id       = ep->dest_addr.id;
    req.timeout_ms       = iface->config.timeout_ms;
    req.private_data     = hdr;
    req.private_data_len = total_len;
    req.max_cm_retries   = iface->config.retry_count;

    /* Create temporary ID for this message. Will be released when getting REP. */
    ret = ib_cm_create_id(iface->cmdev, &id, NULL);
    if (ret) {
        ucs_error("ib_cm_create_id() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_free;
    }

    uct_cm_dump_path(&path);

    ret = ib_cm_send_sidr_req(id, &req);
    if (ret) {
        ucs_error("ib_cm_send_sidr_req() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_destroy_id;
    }

    uct_cm_iface_trace_data(iface, UCT_AM_TRACE_TYPE_SEND, hdr,
                            "TX: SIDR_REQ [dlid %d svc 0x%"PRIx64"]",
                            ntohs(path.dlid), req.service_id);
    ucs_free(hdr);
    return payload_len;

err_destroy_id:
    ib_cm_destroy_id(id);
err_free:
    ucs_free(hdr);
err_dec_outstanding:
    ucs_atomic_add32(&iface->outstanding, -1);
    return status;
}

ucs_status_t uct_cm_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *req)
{
    uct_cm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cm_iface_t);
    uct_cm_ep_t *ep = ucs_derived_of(tl_ep, uct_cm_ep_t);

    UCS_ASYNC_BLOCK(iface->super.super.worker->async);
    ucs_derived_of(uct_pending_req_priv(req), uct_cm_pending_req_priv_t)->ep = ep;
    uct_pending_req_push(&iface->notify_q, req);
    UCS_ASYNC_UNBLOCK(iface->super.super.worker->async);
    return UCS_OK;
}

void uct_cm_ep_pending_purge(uct_ep_h tl_ep, uct_pending_callback_t cb)
{
    uct_cm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cm_iface_t);
    uct_cm_ep_t *ep = ucs_derived_of(tl_ep, uct_cm_ep_t);
    uct_cm_pending_req_priv_t *priv;

    uct_pending_queue_purge(priv, &iface->notify_q, priv->ep == ep, cb);
}

ucs_status_t uct_cm_ep_flush(uct_ep_h tl_ep)
{
    return uct_cm_iface_flush(tl_ep->iface);
}
