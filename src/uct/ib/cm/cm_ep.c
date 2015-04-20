/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) 2007-2009 Cisco Systems, Inc.  All rights reserved.
* Copyright (c) 2009      IBM Corporation.  All rights reserved.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "cm.h"

#include <infiniband/arch.h>
#include <uct/tl/tl_log.h>

typedef struct uct_cm_iov {
    uct_pack_callback_t pack;
    const void          *arg;
    size_t              length;
} uct_cm_iov_t;


static UCS_CLASS_INIT_FUNC(uct_cm_ep_t, uct_iface_t *tl_iface, const struct sockaddr *addr)

{
    uct_cm_iface_t *iface = ucs_derived_of(tl_iface, uct_cm_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);
    self->dest_addr = *(const uct_sockaddr_ib_t*)addr;
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
    uint16_t pkey;
    int ret;

    ret = ibv_query_pkey(uct_ib_iface_device(&iface->super)->ibv_context,
                         iface->super.port_num, 0 /*TODO config */, &pkey);
    if (ret) {
        ucs_error("ibv_query_pkey() failed: %m");
        return UCS_ERR_INVALID_ADDR;
    }

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
    path->pkey                      = ntohs(pkey);
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
                   path->slid, sgid_buf, path->dlid, dgid_buf);
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

static ucs_status_t uct_cm_ep_send(uct_cm_ep_t *ep, uct_cm_iov_t *iov, int iovcnt)
{
    uct_cm_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_cm_iface_t);
    struct ib_cm_sidr_req_param req;
    struct ibv_sa_path_rec path;
    struct ib_cm_id *id;
    ucs_status_t status;
    size_t length, offset;
    void *buffer;
    int ret;
    int i;

    ucs_trace_func("");

    /* Count total length */
    length = 0;
    for (i = 0; i < iovcnt; ++i) {
        length += iov[i].length;
    }
    if (length > IB_CM_SIDR_REQ_PRIVATE_DATA_SIZE) {
        ucs_error("Data too large for SIDR (got: %zu, max: %zu)",
                  length, (size_t)IB_CM_SIDR_REQ_PRIVATE_DATA_SIZE);
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    if (ucs_atomic_fadd32(&iface->inflight, +1) >= 1) {
        status = UCS_ERR_NO_RESOURCE;
        goto err_dec_inflight;
    }

    /* Allocate temporary contiguous buffer */
    buffer = ucs_malloc(length, "cm_send_buf");
    if (buffer == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_dec_inflight;
    }

    /* Copy IOV data to buffer */
    offset = 0;
    for (i = 0; i < iovcnt; ++i) {
        iov[i].pack(buffer + offset, (void*)iov[i].arg, iov[i].length);
        offset += iov[i].length;
    }

    status = uct_cm_ep_fill_path_rec(ep, &path);
    if (status != UCS_OK) {
        goto err_free;
    }

    /* Fill SIDR request */
    memset(&req, 0, sizeof req);
    req.path             = &path;
    req.service_id       = ep->dest_addr.id;
    req.timeout_ms       = iface->config.timeout_ms;
    req.private_data     = buffer;
    req.private_data_len = length;
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

    ucs_trace_data("SEND SIDR_REQ [dlid %d service_id 0x%"PRIx64"] am_id %d",
                   ntohs(path.dlid), req.service_id, ((uct_cm_hdr_t*)buffer)->am_id);
    ucs_free(buffer);
    return UCS_OK;

err_destroy_id:
    ib_cm_destroy_id(id);
err_free:
    ucs_free(buffer);
err_dec_inflight:
    ucs_atomic_add32(&iface->inflight, -1);
err:
    return status;
}

ucs_status_t uct_cm_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t header,
                                const void *payload, unsigned length)
{
    uct_cm_ep_t *ep = ucs_derived_of(tl_ep, uct_cm_ep_t);
    ucs_status_t status;
    uct_cm_iov_t iov[3];
    uct_cm_hdr_t hdr;

    UCT_CHECK_AM_ID(id);

    hdr.am_id       = id;
    hdr.length      = sizeof(header) + length;
    iov[0].pack     = (uct_pack_callback_t)memcpy;
    iov[0].arg      = &hdr;
    iov[0].length   = sizeof(hdr);
    iov[1].pack     = (uct_pack_callback_t)memcpy;
    iov[1].arg      = &header;
    iov[1].length   = sizeof(header);
    iov[2].pack     = (uct_pack_callback_t)memcpy;
    iov[2].arg      = payload;
    iov[2].length   = length;

    status = uct_cm_ep_send(ep, iov, 3);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super, AM, SHORT,
                                 sizeof(header) + length);
    return status;
}

ucs_status_t uct_cm_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                uct_pack_callback_t pack_cb, void *arg,
                                size_t length)
{
    uct_cm_ep_t *ep = ucs_derived_of(tl_ep, uct_cm_ep_t);
    ucs_status_t status;
    uct_cm_iov_t iov[3];
    uct_cm_hdr_t hdr;

    UCT_CHECK_AM_ID(id);

    hdr.am_id       = id;
    hdr.length      = length;
    iov[0].pack     = (uct_pack_callback_t)memcpy;
    iov[0].arg      = &hdr;
    iov[0].length   = sizeof(hdr);
    iov[1].pack     = pack_cb;
    iov[1].arg      = arg;
    iov[1].length   = length;

    status = uct_cm_ep_send(ep, iov, 2);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super, AM, BCOPY, length);
    return status;
}

ucs_status_t uct_cm_ep_req_notify(uct_ep_h tl_ep, uct_completion_t *comp)
{
    uct_cm_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cm_iface_t);
    uct_cm_completion_t *cm_comp = ucs_derived_of(comp, uct_cm_completion_t);

    ucs_queue_push(&iface->notify, &cm_comp->queue);
    return UCS_OK;
}

ucs_status_t uct_cm_ep_flush(uct_ep_h tl_ep)
{
    return uct_cm_iface_flush(tl_ep->iface);
}
