/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rma.h"
#include "rma.inl"

#include <ucs/arch/atomic.h>
#include <ucs/profile/profile.h>


static size_t ucp_amo_sw_pack(void *dest, void *arg, uint8_t fetch)
{
    ucp_request_t *req            = arg;
    ucp_atomic_req_hdr_t *atomich = dest;
    ucp_ep_t *ep                  = req->send.ep;
    size_t size                   = req->send.length;
    size_t length;

    atomich->address    = req->send.rma.remote_addr;
    atomich->req.ep_ptr = ucp_ep_dest_ep_ptr(ep);
    atomich->req.reqptr = fetch ? (uintptr_t)req : 0;
    atomich->length     = size;
    atomich->opcode     = req->send.amo.uct_op;

    memcpy(atomich + 1, &req->send.amo.value, size);
    length = sizeof(*atomich) + size;

    if (req->send.amo.uct_op == UCT_ATOMIC_OP_CSWAP) {
        /* compare-swap has two arguments */
        memcpy(UCS_PTR_BYTE_OFFSET(atomich + 1, size), req->send.buffer, size);
        length += size;
    }

    return length;
}

static size_t ucp_amo_sw_post_pack_cb(void *dest, void *arg)
{
    return ucp_amo_sw_pack(dest, arg, 0);
}

static size_t ucp_amo_sw_fetch_pack_cb(void *dest, void *arg)
{
    return ucp_amo_sw_pack(dest, arg, 1);
}

static ucs_status_t ucp_amo_sw_progress(uct_pending_req_t *self,
                                        uct_pack_callback_t pack_cb, int fetch)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ucs_status_t status;
    ssize_t packed_len;

    req->send.lane = ucp_ep_get_am_lane(ep);
    packed_len = uct_ep_am_bcopy(ep->uct_eps[req->send.lane],
                                 UCP_AM_ID_ATOMIC_REQ, pack_cb, req, 0);
    if (packed_len > 0) {
        ucp_ep_rma_remote_request_sent(ep);
        if (!fetch) {
            ucp_request_complete_send(req, UCS_OK);
        }
        return UCS_OK;
    } else {
        status = (ucs_status_t)packed_len;
        if (status != UCS_ERR_NO_RESOURCE) {
            /* failure */
            ucp_request_complete_send(req, status);
        }
        return status;
    }
}

static ucs_status_t ucp_amo_sw_progress_post(uct_pending_req_t *self)
{
    return ucp_amo_sw_progress(self, ucp_amo_sw_post_pack_cb, 0);
}

static ucs_status_t ucp_amo_sw_progress_fetch(uct_pending_req_t *self)
{
    return ucp_amo_sw_progress(self, ucp_amo_sw_fetch_pack_cb, 1);
}

ucp_amo_proto_t ucp_amo_sw_proto = {
    .name           = "sw_amo",
    .progress_fetch = ucp_amo_sw_progress_fetch,
    .progress_post  = ucp_amo_sw_progress_post
};

static size_t ucp_amo_sw_pack_atomic_reply(void *dest, void *arg)
{
    ucp_rma_rep_hdr_t *hdr = dest;
    ucp_request_t *req     = arg;

    hdr->req = req->send.get_reply.req;

    switch (req->send.length) {
    case sizeof(uint32_t):
        *(uint32_t*)(hdr + 1) = req->send.atomic_reply.data.reply32;
        break;
    case sizeof(uint64_t):
        *(uint64_t*)(hdr + 1) = req->send.atomic_reply.data.reply64;
        break;
    default:
        ucs_fatal("invalid atomic length: %zu", req->send.length);
    }

    return sizeof(*hdr) + req->send.length;
}

static ucs_status_t ucp_progress_atomic_reply(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ssize_t packed_len;

    req->send.lane = ucp_ep_get_am_lane(ep);
    packed_len = uct_ep_am_bcopy(ep->uct_eps[req->send.lane], UCP_AM_ID_ATOMIC_REP,
                                 ucp_amo_sw_pack_atomic_reply, req, 0);

    if (packed_len < 0) {
        return (ucs_status_t)packed_len;
    }

    ucs_assert(packed_len == sizeof(ucp_rma_rep_hdr_t) + req->send.length);
    ucp_request_put(req);
    return UCS_OK;
}

#define DEFINE_AMO_SW_OP(_bits) \
    static void ucp_amo_sw_do_op##_bits(const ucp_atomic_req_hdr_t *atomicreqh) \
    { \
        uint##_bits##_t *ptr  = (void*)atomicreqh->address; \
        uint##_bits##_t *args = (void*)(atomicreqh + 1); \
        \
       switch (atomicreqh->opcode) { \
        case UCT_ATOMIC_OP_ADD: \
            ucs_atomic_add##_bits(ptr, args[0]); \
            break; \
        case UCT_ATOMIC_OP_AND: \
            ucs_atomic_and##_bits(ptr, args[0]); \
            break; \
        case UCT_ATOMIC_OP_OR: \
            ucs_atomic_or##_bits(ptr, args[0]); \
            break; \
        case UCT_ATOMIC_OP_XOR: \
            ucs_atomic_xor##_bits(ptr, args[0]); \
            break; \
        default: \
            ucs_fatal("invalid opcode: %d", atomicreqh->opcode); \
        } \
    }

#define DEFINE_AMO_SW_FOP(_bits) \
    static void ucp_amo_sw_do_fop##_bits(const ucp_atomic_req_hdr_t *atomicreqh, \
                                         ucp_atomic_reply_t *result) \
    { \
        uint##_bits##_t *ptr  = (void*)atomicreqh->address; \
        uint##_bits##_t *args = (void*)(atomicreqh + 1); \
        \
        switch (atomicreqh->opcode) { \
        case UCT_ATOMIC_OP_ADD: \
            result->reply##_bits = ucs_atomic_fadd##_bits(ptr, args[0]); \
            break; \
        case UCT_ATOMIC_OP_AND: \
            result->reply##_bits = ucs_atomic_fand##_bits(ptr, args[0]); \
            break; \
        case UCT_ATOMIC_OP_OR: \
            result->reply##_bits = ucs_atomic_for##_bits(ptr, args[0]); \
            break; \
        case UCT_ATOMIC_OP_XOR: \
            result->reply##_bits = ucs_atomic_fxor##_bits(ptr, args[0]); \
            break; \
        case UCT_ATOMIC_OP_SWAP: \
            result->reply##_bits = ucs_atomic_swap##_bits(ptr, args[0]); \
            break; \
        case UCT_ATOMIC_OP_CSWAP: \
            result->reply##_bits = ucs_atomic_cswap##_bits(ptr, args[0], args[1]); \
            break; \
        default: \
            ucs_fatal("invalid opcode: %d", atomicreqh->opcode); \
        } \
    }

DEFINE_AMO_SW_OP(32)
DEFINE_AMO_SW_OP(64)
DEFINE_AMO_SW_FOP(32)
DEFINE_AMO_SW_FOP(64)

typedef struct ucp_atomic_flush_ctx {
    ucp_atomic_req_hdr_t req_hdr;
    uint64_t             value;
    ucp_mem_h            memh;
    ucp_rkey_h           rkey;
} ucp_atomic_flush_ctx_t;

static void ucp_amo_sw_loopback_send_reply(ucp_worker_h worker,
                                           ucp_atomic_flush_ctx_t *flush_ctx)
{
    ucp_ep_h reply_ep;
    ucp_request_t *req;
    ucs_status_t status;

    reply_ep = ucp_worker_get_ep_by_ptr(worker, flush_ctx->req_hdr.req.ep_ptr);

    if (flush_ctx->req_hdr.req.reqptr == 0) {
        ucp_rma_sw_send_cmpl(reply_ep);
    } else {
        req = ucp_request_get(worker);
        if (req == NULL) {
            ucs_error("failed to allocate atomic reply");
        } else {
            req->send.ep               = reply_ep;
            req->send.atomic_reply.req = flush_ctx->req_hdr.req.reqptr;
            req->send.length           = flush_ctx->req_hdr.length;
            req->send.uct.func         = ucp_progress_atomic_reply;
            ucp_request_send(req, 0);
        }
    }

    ucp_rkey_destroy(flush_ctx->rkey);
    status = ucp_mem_unmap(worker->context, flush_ctx->memh);

    if (status != UCS_OK) {
        ucs_error("failed to unmap memory handle %p with error %s",
                  flush_ctx->memh, ucs_status_string(status));
    }

    ucs_free(flush_ctx);
}

static void ucp_amo_sw_loopback_flush_cb(ucp_request_t *req)
{
    ucp_atomic_flush_ctx_t *flush_ctx = req->send.flush_ctx;
    ucp_worker_h worker               = req->send.ep->worker;

    ucp_amo_sw_loopback_send_reply(worker, flush_ctx);
    ucp_request_put(req);
}

static ucs_status_t
ucp_amo_sw_loopback_post(ucp_worker_h worker, ucp_atomic_req_hdr_t *atomicreqh)
{
    ucp_atomic_flush_ctx_t *flush_ctx;
    ucp_mem_map_params_t params;
    void *rkey_buffer;
    size_t rkey_buffer_size;
    void *request;
    ucs_status_t status, status_unmap;

    /* TODO: optimization: can use `am_flags & UCT_CB_PARAM_FLAG_DESC` to avoid
     *       extra allocation and copy for some transports */
    flush_ctx = ucs_malloc(sizeof(*flush_ctx), "loop back amo flush ctx");
    if (flush_ctx == NULL) {
        ucs_error("failed to allocate loopback AMO flush context");
        return UCS_ERR_NO_MEMORY;
    }

    flush_ctx->req_hdr = *atomicreqh;
    flush_ctx->value   = *(uint64_t *)(atomicreqh + 1);

    params.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                         UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                         UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.address     = (void *)atomicreqh->address;
    params.length      = atomicreqh->length;
    params.flags       = 0;

    /*
     * TODO: rkey can be passed as a part of AM to avoid
     *       mem_map/rkey_pack/rkey_pack.
     * NOTE: if rcache is enabled then registration should be "for free" since
     *       destination memory has been registered.
     */
    status = ucp_mem_map(worker->context, &params, &flush_ctx->memh);
    if (status != UCS_OK) {
        ucs_error("failed to map memory region %p length %zu status %s",
                  params.address, params.length, ucs_status_string(status));
        goto err_free_ctx;
    }

    status = ucp_rkey_pack(worker->context, flush_ctx->memh, &rkey_buffer,
                           &rkey_buffer_size);
    if (status != UCS_OK) {
        ucs_error("failed to pack rkey for region %p length %zu, status %s",
                  params.address, params.length, ucs_status_string(status));
        goto err_unmap;
    }

    status = ucp_ep_rkey_unpack(worker->atomic_ep, rkey_buffer,
                                &flush_ctx->rkey);
    ucp_rkey_buffer_release(rkey_buffer);
    if (status != UCS_OK) {
        ucs_error("failed to unpack rkey for region %p length %zu, status %s",
                  params.address, params.length, ucs_status_string(status));
        goto err_unmap;
    }

    status = ucp_atomic_post(worker->atomic_ep,
                             (ucp_atomic_post_op_t)atomicreqh->opcode,
                             flush_ctx->value, atomicreqh->length,
                             atomicreqh->address, flush_ctx->rkey);
    if (status != UCS_OK) {
        ucs_error("failed to post loopback AMO to ep %p, status %s",
                  worker->atomic_ep, ucs_status_string(status));
        goto err_unmap;
    }

    request = ucp_ep_flush_internal(worker->atomic_ep, UCT_FLUSH_FLAG_LOCAL,
                                    NULL, 0, NULL, flush_ctx,
                                    ucp_amo_sw_loopback_flush_cb,
                                    "sw amo loopback");
    if (request == NULL) {
        ucp_amo_sw_loopback_send_reply(worker, flush_ctx);
        return UCS_OK;
    }

    if (UCS_PTR_IS_PTR(request)) {
        return UCS_OK;
    }

    ucs_error("failed to flush loopback AMO to ep %p, status %s",
              worker->atomic_ep, ucs_status_string(status));
    status = UCS_PTR_STATUS(request);

err_unmap:
    status_unmap = ucp_mem_unmap(worker->context, flush_ctx->memh);
    if (status_unmap != UCS_OK) {
        ucs_warn("failed to unmap %p memory handler, %s", flush_ctx->memh,
                 ucs_status_string(status_unmap));
    }

err_free_ctx:
    ucs_free(flush_ctx);
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t,
                 ucp_atomic_req_handler, (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    ucp_atomic_req_hdr_t *atomicreqh = data;
    ucp_worker_h worker              = arg;
    ucp_ep_h ep                      = ucp_worker_get_ep_by_ptr(worker,
                                                                atomicreqh->req.ep_ptr);
    ucp_rsc_index_t amo_rsc_idx      = ucs_ffs64_safe(worker->atomic_tls);
    ucp_request_t *req;

    if (ucs_unlikely((amo_rsc_idx != UCP_MAX_RESOURCES) &&
                     (ucp_worker_iface_get_attr(worker,
                                                amo_rsc_idx)->cap.flags &
                      UCT_IFACE_FLAG_ATOMIC_DEVICE))) {
        return ucp_amo_sw_loopback_post(worker, atomicreqh);
    }

    if (atomicreqh->req.reqptr == 0) {
        /* atomic operation without result */
        switch (atomicreqh->length) {
        case sizeof(uint32_t):
            ucp_amo_sw_do_op32(atomicreqh);
            break;
        case sizeof(uint64_t):
            ucp_amo_sw_do_op64(atomicreqh);
            break;
        default:
            ucs_fatal("invalid atomic length: %u", atomicreqh->length);
        }
        ucp_rma_sw_send_cmpl(ep);
    } else {
        /* atomic operation with result */
        req = ucp_request_get(worker);
        if (req == NULL) {
            ucs_error("failed to allocate atomic reply");
            return UCS_OK;
        }

        switch (atomicreqh->length) {
        case sizeof(uint32_t):
            ucp_amo_sw_do_fop32(atomicreqh, &req->send.atomic_reply.data);
            break;
        case sizeof(uint64_t):
            ucp_amo_sw_do_fop64(atomicreqh, &req->send.atomic_reply.data);
            break;
        default:
            ucs_fatal("invalid atomic length: %u", atomicreqh->length);
        }

        req->send.ep               = ep;
        req->send.atomic_reply.req = atomicreqh->req.reqptr;
        req->send.length           = atomicreqh->length;
        req->send.uct.func         = ucp_progress_atomic_reply;
        ucp_request_send(req, 0);
    }

    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_rep_handler, (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    ucp_rma_rep_hdr_t *hdr = data;
    size_t frag_length     = length - sizeof(*hdr);
    ucp_request_t *req     = (ucp_request_t*)hdr->req;
    ucp_ep_h ep            = req->send.ep;

    memcpy(req->send.buffer, hdr + 1, frag_length);
    ucp_request_complete_send(req, UCS_OK);
    ucp_ep_rma_remote_request_completed(ep);
    return UCS_OK;
}

static void ucp_amo_sw_dump_packet(ucp_worker_h worker, uct_am_trace_type_t type,
                                   uint8_t id, const void *data, size_t length,
                                   char *buffer, size_t max)
{
    const ucp_atomic_req_hdr_t *atomich;
    const ucp_rma_rep_hdr_t *reph;
    size_t header_len;
    char *p;

    switch (id) {
    case UCP_AM_ID_ATOMIC_REQ:
        atomich = data;
        snprintf(buffer, max,
                 "ATOMIC_REQ [addr 0x%lx len %u reqptr 0x%lx ep 0x%lx op %d]",
                 atomich->address, atomich->length, atomich->req.reqptr,
                 atomich->req.ep_ptr, atomich->opcode);
        header_len = sizeof(*atomich);;
        break;
    case UCP_AM_ID_ATOMIC_REP:
        reph = data;
        snprintf(buffer, max, "ATOMIC_REP [reqptr 0x%lx]", reph->req);
        header_len = sizeof(*reph);
        break;
    default:
        return;
    }

    p = buffer + strlen(buffer);
    ucp_dump_payload(worker->context, p, buffer + max - p,
                     UCS_PTR_BYTE_OFFSET(data, header_len),
                     length - header_len);
}

UCP_DEFINE_AM(UCP_FEATURE_AMO, UCP_AM_ID_ATOMIC_REQ, ucp_atomic_req_handler,
              ucp_amo_sw_dump_packet, 0);
UCP_DEFINE_AM(UCP_FEATURE_AMO, UCP_AM_ID_ATOMIC_REP, ucp_atomic_rep_handler,
              ucp_amo_sw_dump_packet, 0);

UCP_DEFINE_AM_PROXY(UCP_AM_ID_ATOMIC_REQ);
