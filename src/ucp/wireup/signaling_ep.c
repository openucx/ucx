/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "wireup.h"

#include <ucp/core/ucp_proxy_ep.h>


/* Context for packing short data into bcopy */
typedef struct {
    uint64_t     header;
    const void   *payload;
    unsigned     length;
} ucp_signaling_ep_pack_ctx_t;


static size_t ucp_signaling_ep_pack_short(void *dest, void *arg)
{
    ucp_signaling_ep_pack_ctx_t *ctx = arg;

    *(uint64_t*)dest = ctx->header;
    memcpy(dest + sizeof(uint64_t), ctx->payload, ctx->length);
    return sizeof(uint64_t) + ctx->length;
}

static ucs_status_t
ucp_signaling_ep_am_short(uct_ep_h ep, uint8_t id, uint64_t header,
                          const void *payload, unsigned length)
{
    ucp_proxy_ep_t *proxy_ep = ucs_derived_of(ep, ucp_proxy_ep_t);
    ucp_signaling_ep_pack_ctx_t ctx;
    ssize_t packed_size;

    ctx.header  = header;
    ctx.payload = payload;
    ctx.length  = length;

    packed_size = uct_ep_am_bcopy(proxy_ep->uct_ep, id,
                                  ucp_signaling_ep_pack_short, &ctx,
                                  UCT_SEND_FLAG_SIGNALED);
    if (packed_size < 0) {
        return (ucs_status_t)packed_size;
    }

    ucp_proxy_ep_replace(proxy_ep);
    return UCS_OK;
}

static ssize_t
ucp_signaling_ep_am_bcopy(uct_ep_h ep, uint8_t id, uct_pack_callback_t pack_cb,
                          void *arg, unsigned flags)
{
    ucp_proxy_ep_t *proxy_ep = ucs_derived_of(ep, ucp_proxy_ep_t);
    ssize_t packed_size;

    packed_size = uct_ep_am_bcopy(proxy_ep->uct_ep, id, pack_cb, arg,
                                  flags | UCT_SEND_FLAG_SIGNALED);
    if (packed_size >= 0) {
        ucp_proxy_ep_replace(proxy_ep);
    }
    return packed_size;
}

static ucs_status_t
ucp_signaling_ep_am_zcopy(uct_ep_h ep, uint8_t id, const void *header,
                          unsigned header_length, const uct_iov_t *iov,
                          size_t iovcnt, unsigned flags, uct_completion_t *comp)
{
    ucp_proxy_ep_t *proxy_ep = ucs_derived_of(ep, ucp_proxy_ep_t);
    ucs_status_t status;

    status = uct_ep_am_zcopy(proxy_ep->uct_ep, id, header, header_length, iov,
                             iovcnt, flags | UCT_SEND_FLAG_SIGNALED, comp);
    if ((status == UCS_OK) || (status == UCS_INPROGRESS)) {
        ucp_proxy_ep_replace(proxy_ep);
    }
    return status;
}

ucs_status_t ucp_signaling_ep_create(ucp_ep_h ucp_ep, uct_ep_h uct_ep,
                                     int is_owner, uct_ep_h *signaling_ep)
{
    static uct_iface_ops_t signaling_ep_ops = {
        .ep_am_short = ucp_signaling_ep_am_short,
        .ep_am_bcopy = ucp_signaling_ep_am_bcopy,
        .ep_am_zcopy = ucp_signaling_ep_am_zcopy
    };

    return UCS_CLASS_NEW(ucp_proxy_ep_t, signaling_ep, &signaling_ep_ops,
                         ucp_ep, uct_ep, is_owner);
}
