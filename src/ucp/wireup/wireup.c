/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "wireup.h"
#include "address.h"
#include "stub_ep.h"

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/dt/dt_contig.h>
#include <ucp/tag/eager.h>
#include <ucs/arch/atomic.h>
#include <ucs/arch/bitops.h>
#include <ucs/async/async.h>
#include <ucs/datastruct/queue.h>


/*
 *  Description of wire-up protocol
 * ==================================
 *
 *   The goal is to expose one-sided connection establishment semantics in UCP
 * layer, and overcome transport and protocol limitations:
 *   (1) Some transports require two-sided, point-to-point connection.
 *   (2) Some protocols require sending replies, which requires creating ucp endpoint
 *       on the remote side as well (even if the user did not create it explicitly
 *       by calling ucp_ep_create).
 *   (3) TBD allow creating multiple endpoints between same pair of workers.
 *
 *  Wire-up process:
 *    1. Select the transport that would be used in runtime, based on required
 *       features (passed to ucp_init), transport capabilities, and performance
 *       estimations.
 *
 *    2. If the selected transport cannot create ep-to-iface connection, select
 *       an "auxiliary" transport to use for wire-up protocol. Then, use a 3-way
 *       handshake protocol (REQ->REP->ACK) to create ep on the remote side and
 *       connect it to local ep. Until this is completed, create a stub uct_ep
 *       whose send functions always return NO_RESOURCE. When the connection is
 *       ready, the stub ep is replaced by the real uct ep.
 *
 *       If the selected transport is capable of ep-to-iface connection, simply
 *       create the connected ep.
 *
 *
 *    3. TBD When we start a protocol which requires remote replies (such as
 *       rendezvous), first need to check if the remote side is also connected
 *       to us. If not, need to start the same 3-way handshake protocol to let
 *       it create the reverse connection. This can be either in "half-handshake"
 *       mode (i.e send the data immediately after sending the connect request)
 *       or in "full-handshake" mode (i.e send the data only after getting a reply).
 */


/**
 * Calculates a score of specific wireup.
 */
typedef double (*ucp_wireup_score_function_t)(ucp_worker_h worker,
                                              uct_iface_attr_t *iface_attr,
                                              char *reason, size_t max);


static void ucp_wireup_stop_aux(ucp_ep_h ep);

static void ucp_wireup_ep_ready_to_send(ucp_ep_h ep)
{
    ep->state |= UCP_EP_STATE_READY_TO_SEND;
    if (ep->state & UCP_EP_STATE_NEXT_EP_REMOTE_CONNECTED) {
        ep->state |= UCP_EP_STATE_READY_TO_RECEIVE;
    }

    ucs_debug("ready to send%s to %s 0x%"PRIx64"->0x%"PRIx64,
              (ep->state & UCP_EP_STATE_READY_TO_RECEIVE) ? " and receive" : "",
              ucp_ep_peer_name(ep), ep->worker->uuid, ep->dest_uuid);
}

void ucp_wireup_progress(ucp_ep_h ep)
{
    ucp_stub_ep_t *stub_ep = ucp_ep_get_stub_ep(ep);
    ucs_queue_head_t tmp_pending_queue;
    ucp_worker_h worker = ep->worker;
    uct_pending_req_t *req;
    ucs_status_t status;
    uct_ep_h uct_ep;

    /*
     * We switch the endpoint in this function (instead in wireup code) since
     * this is guaranteed to run from the main thread.
     * Don't start using the transport before the wireup protocol finished
     * sending ack/reply.
     */
    sched_yield();
    ucs_async_check_miss(&ep->worker->async);

    /*
     * Check that we are ready to switch:
     * - Remote side must also be connected.
     * - We should have sent a wireup reply to remote side
     * - We should have sent all pending wireup operations (so we won't discard them)
     */
    if (!(ep->state & UCP_EP_STATE_NEXT_EP) ||
        !(ep->state & UCP_EP_STATE_NEXT_EP_REMOTE_CONNECTED) ||
        !(ep->state & (UCP_EP_STATE_WIREUP_REPLY_SENT|UCP_EP_STATE_WIREUP_ACK_SENT)) ||
        (stub_ep->pending_count != 0))
    {
        return;
    }

    ucs_memory_cpu_fence();
    UCS_ASYNC_BLOCK(&worker->async);

    /* Take out next_ep */
    uct_ep = stub_ep->next_ep;
    ucs_assert(uct_ep != NULL);
    stub_ep->next_ep = NULL;
    ep->state &= ~UCP_EP_STATE_NEXT_EP;

    /* Move stub pending queue to temporary queue and remove references to
     * the stub progress function
     */
    ucs_queue_head_init(&tmp_pending_queue);
    ucs_queue_for_each_extract(req, &stub_ep->pending_q, priv, 1) {
        uct_worker_progress_unregister(ep->worker->uct, ucp_stub_ep_progress,
                                       stub_ep);
        ucs_queue_push(&tmp_pending_queue, ucp_stub_ep_req_priv(req));
    }

    /* Destroy temporary endpoints */
    stub_ep = NULL;
    ucp_wireup_stop_aux(ep);

    /* Switch to real transport */
    ep->uct_ep = uct_ep;
    ucp_wireup_ep_ready_to_send(ep);

    UCS_ASYNC_UNBLOCK(&worker->async);

    /* Replay pending requests */
    ucs_queue_for_each_extract(req, &tmp_pending_queue, priv, 1) {
        do {
            status = ucp_ep_add_pending_uct(ep, uct_ep, req);
        } while (status != UCS_OK);
        --ep->worker->stub_pend_count;
    }
}

static double ucp_aux_score_func(ucp_worker_h worker, uct_iface_attr_t *iface_attr,
                                 char *reason, size_t max)
{
    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_BCOPY)) {
        strncpy(reason, "am_bcopy for wireup", max);
        return 0.0;
    }

    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE)) {
        strncpy(reason, "connecting to iface", max);
        return 0.0;
    }

    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE)) {
        strncpy(reason, "async am callback", max);
        return 0.0;
    }

    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE)) {
        strncpy(reason, "pending", max);
        return 0.0;
    }

    return (1e-3 / iface_attr->latency) +
           (1e3 * ucs_max(iface_attr->cap.am.max_bcopy, iface_attr->cap.am.max_short));
}

static double ucp_runtime_score_func(ucp_worker_h worker, uct_iface_attr_t *iface_attr,
                                     char *reason, size_t max)
{
    ucp_context_t *context = worker->context;

    if (iface_attr->cap.flags & UCT_IFACE_FLAG_AM_DUP) {
        strncpy(reason, "full reliability", max);
        return 0.0;
    }

    if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
        if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_BCOPY)) {
            strncpy(reason, "am_bcopy for wireup", max);
            return 0.0;
        }
    }

    if (context->config.features & UCP_FEATURE_TAG) {
        if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_SHORT)) {
            strncpy(reason, "am_short for tag", max);
            return 0.0;
        }
        if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_PENDING)) {
            strncpy(reason, "pending", max);
            return 0.0;
        }
        if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_CB_SYNC)) {
            strncpy(reason, "sync am callback for tag", max);
            return 0.0;
        }
    }

    if (context->config.features & UCP_FEATURE_RMA) {
        /* TODO remove this requirement once we have RMA emulation */
        if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_PUT_SHORT)) {
            strncpy(reason, "put_short for rma", max);
            return 0.0;
        }
        if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_PUT_BCOPY)) {
            strncpy(reason, "put_bcopy for rma", max);
            return 0.0;
        }
        if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_GET_BCOPY)) {
            strncpy(reason, "get_bcopy for rma", max);
            return 0.0;
        }
    }

    if (context->config.features & UCP_FEATURE_AMO32) {
        /* TODO remove this requirement once we have SW atomics */
        if (!ucs_test_all_flags(iface_attr->cap.flags,
                                UCT_IFACE_FLAG_ATOMIC_ADD32 |
                                UCT_IFACE_FLAG_ATOMIC_FADD32 |
                                UCT_IFACE_FLAG_ATOMIC_SWAP32 |
                                UCT_IFACE_FLAG_ATOMIC_CSWAP32))
        {
            strncpy(reason, "all 32-bit atomics", max);
            return 0.0;
        }
    }

    if (context->config.features & UCP_FEATURE_AMO64) {
        /* TODO remove this requirement once we have SW atomics */
        if (!ucs_test_all_flags(iface_attr->cap.flags,
                                UCT_IFACE_FLAG_ATOMIC_ADD64 |
                                UCT_IFACE_FLAG_ATOMIC_FADD64 |
                                UCT_IFACE_FLAG_ATOMIC_SWAP64 |
                                UCT_IFACE_FLAG_ATOMIC_CSWAP64))
        {
            strncpy(reason, "all 64-bit atomics", max);
            return 0.0;
        }
    }

    if (context->config.features & UCP_FEATURE_WAKEUP) {
        if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_WAKEUP)) {
            strncpy(reason, "wakeup", max);
            return 0.0;
        }
    }

    return 1e-3 / (iface_attr->latency + (iface_attr->overhead * 2));
}

/**
 * Select a local and remote transport
 */
static ucs_status_t ucp_select_transport(ucp_worker_h worker, const char *peer_name,
                                         const ucp_address_entry_t *address_list,
                                         unsigned address_count,
                                         ucp_rsc_index_t pd_index,
                                         ucp_rsc_index_t *rsc_index_p,
                                         unsigned *dst_addr_index_p,
                                         ucp_wireup_score_function_t score_func,
                                         const char *title)
{
    ucp_context_h context = worker->context;
    uct_tl_resource_desc_t *resource;
    const ucp_address_entry_t *ae;
    ucp_rsc_index_t rsc_index;
    double score, best_score;
    char tls_info[256];
    char tl_reason[64];
    char *p, *endp;
    uct_iface_h iface;
    int reachable;
    int found;

    found      = 0;
    best_score = 0.0;
    p          = tls_info;
    endp       = tls_info + sizeof(tls_info) - 1;
    *endp      = 0;

    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        resource   = &context->tl_rscs[rsc_index].tl_rsc;
        iface      = worker->ifaces[rsc_index];

        /* Must use only the pd the remote side explicitly requested */
        if ((pd_index != UCP_NULL_RESOURCE) &&
            (pd_index != context->tl_rscs[rsc_index].pd_index))
        {
            const char * pd_name = context->pd_rscs[pd_index].pd_name;
            ucs_trace(UCT_TL_RESOURCE_DESC_FMT " : not on pd %s",
                      UCT_TL_RESOURCE_DESC_ARG(resource), pd_name);
            snprintf(p, endp - p, ", "UCT_TL_RESOURCE_DESC_FMT" - not on pd %s",
                     UCT_TL_RESOURCE_DESC_ARG(resource), pd_name);
            p += strlen(p);
            continue;
        }

        /* Get local device score */
        score = score_func(worker, &worker->iface_attrs[rsc_index], tl_reason,
                           sizeof(tl_reason));
        if (score <= 0.0) {
            ucs_trace(UCT_TL_RESOURCE_DESC_FMT " :  not suitable for %s, no %s",
                      UCT_TL_RESOURCE_DESC_ARG(resource), title, tl_reason);
            snprintf(p, endp - p, ", "UCT_TL_RESOURCE_DESC_FMT" - no %s",
                     UCT_TL_RESOURCE_DESC_ARG(resource), tl_reason);
            p += strlen(p);
            continue;
        }

        /* Check if remote peer is reachable using one of its devices */
        reachable = 0;
        for (ae = address_list; ae < address_list + address_count; ++ae) {
            /* Must be reachable device address, on same transport */
            reachable = !strcmp(ae->tl_name, resource->tl_name) &&
                         uct_iface_is_reachable(iface, ae->dev_addr);
            if (reachable) {
                break;
            }
        }
        if (!reachable) {
            ucs_trace(UCT_TL_RESOURCE_DESC_FMT " : cannot reach to %s",
                      UCT_TL_RESOURCE_DESC_ARG(resource), peer_name);
            snprintf(p, endp - p, ", "UCT_TL_RESOURCE_DESC_FMT" - unreachable",
                     UCT_TL_RESOURCE_DESC_ARG(resource));
            p += strlen(p);
            continue;
        }

        ucs_trace(UCT_TL_RESOURCE_DESC_FMT " : %s score %.2f",
                  UCT_TL_RESOURCE_DESC_ARG(resource), title, score);
        if (!found || (score > best_score)) {
            found             = 1;
            best_score        = score;
            *rsc_index_p      = rsc_index;
            *dst_addr_index_p = ae - address_list;
        }
    }

    if (!found) {
        ucs_error("No suitable %s transport to %s: %s", title, peer_name,
                  tls_info + 2);
        return UCS_ERR_UNREACHABLE;
    }

    ucs_debug("selected for %s: " UCT_TL_RESOURCE_DESC_FMT " -> %s address[%d] score %.2f",
              title, UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[*rsc_index_p].tl_rsc),
              peer_name, *dst_addr_index_p, best_score);
    return UCS_OK;
}

static void ucp_wireup_msg_dump(ucp_worker_h worker, uct_am_trace_type_t type,
                                uint8_t id, const void *data, size_t length,
                                char *buffer, size_t max)
{
    const ucp_wireup_msg_t *msg = data;
    char peer_name[UCP_WORKER_NAME_MAX + 1];
    ucp_address_entry_t *address_list, *ae;
    unsigned address_count;
    uint64_t uuid;
    char *p, *end;

    ucp_address_unpack(msg + 1, &uuid, peer_name, sizeof(peer_name),
                       &address_count, &address_list);

    p   = buffer;
    end = buffer + max;
    snprintf(p, end - p, "WIREUP %s [%s uuid 0x%"PRIx64" -> pd %d]",
             (msg->type == UCP_WIREUP_MSG_REQUEST ) ? "REQ" :
             (msg->type == UCP_WIREUP_MSG_REPLY   ) ? "REP" :
             (msg->type == UCP_WIREUP_MSG_ACK     ) ? "ACK" : "",
             peer_name, uuid, msg->dst_pd_index);

    p += strlen(p);
    for (ae = address_list; ae < address_list + address_count; ++ae) {
        snprintf(p, end - p, " [%s(%zu)%s]", ae->tl_name, ae->tl_addr_len,
                 ((ae - address_list) == msg->aux_index) ? " aux" :
                 ((ae - address_list) == msg->tl_index ) ? " tl" :
                 "");
        p += strlen(p);
    }

    ucs_free(address_list);
}

static uct_ep_h ucp_wireup_msg_ep(ucp_ep_h ep)
{
    /* If the transport is fully wired, use it for messages */
    if (ep->state & UCP_EP_STATE_READY_TO_SEND) {
        return ep->uct_ep;
    } else if (ucs_test_all_flags(ep->state, UCP_EP_STATE_NEXT_EP_LOCAL_CONNECTED|
                                  UCP_EP_STATE_NEXT_EP_REMOTE_CONNECTED))
    {
        /* If next_ep is fully wired, use it for messages */
        return ucp_ep_get_stub_ep(ep)->next_ep;
    } else if (ep->state & UCP_EP_STATE_AUX_EP) {
        /* Otherwise we have no choice but to use the auxiliary */
        return ucp_ep_get_stub_ep(ep)->aux_ep;
    }

    ucs_fatal("no valid transport to send wireup message");
}

typedef struct {
    ucp_wireup_msg_t msg;
    void             *address;
    size_t           address_size;
} ucp_wireup_msg_pack_ctx_t;

static size_t ucp_wireup_msg_pack(void *dest, void *arg)
{
    ucp_wireup_msg_pack_ctx_t *ctx = arg;
    *(ucp_wireup_msg_t*)dest = ctx->msg;
    memcpy((ucp_wireup_msg_t*)dest + 1, ctx->address, ctx->address_size);
    return sizeof(ucp_wireup_msg_t) + ctx->address_size;
}

static ucs_status_t ucp_wireup_msg_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_wireup_msg_pack_ctx_t wireup_pack_ctx;
    ucp_ep_h ep = req->send.ep;
    ucp_rsc_index_t rsc_index = ep->rsc_index;
    ucp_rsc_index_t aux_rsc_index = req->send.wireup.aux_rsc_index;
    ucs_status_t status;
    ssize_t packed_len;
    uint64_t tl_bitmap;
    unsigned order[2];

    ucs_assert(ep->dst_pd_index != UCP_NULL_RESOURCE);
    wireup_pack_ctx.msg.type         = req->send.wireup.type;
    wireup_pack_ctx.msg.dst_pd_index = ep->dst_pd_index;

    if (req->send.wireup.type == UCP_WIREUP_MSG_ACK) {
        tl_bitmap = 0;
    } else {
        tl_bitmap = UCS_BIT(rsc_index);
        if (aux_rsc_index != UCP_NULL_RESOURCE) {
            tl_bitmap |= UCS_BIT(aux_rsc_index);
        }
    }

    status = ucp_address_pack(ep->worker, ep, tl_bitmap, order,
                              &wireup_pack_ctx.address_size,
                              &wireup_pack_ctx.address);
    if (status != UCS_OK) {
        ucs_error("failed to pack address: %s", ucs_status_string(status));
        return status;
    }

    if (aux_rsc_index == UCP_NULL_RESOURCE) {
        wireup_pack_ctx.msg.aux_index = -1;
    } else {
        wireup_pack_ctx.msg.aux_index =
                order[ucs_count_one_bits(tl_bitmap & UCS_MASK(aux_rsc_index))];
    }

    if (req->send.wireup.type == UCP_WIREUP_MSG_ACK) {
        wireup_pack_ctx.msg.tl_index = -1;
    } else {
        wireup_pack_ctx.msg.tl_index =
                order[ucs_count_one_bits(tl_bitmap & UCS_MASK(rsc_index))];

        ucs_assertv(wireup_pack_ctx.msg.aux_index != wireup_pack_ctx.msg.tl_index,
                    "aux_index=%d tl_index=%d rsc_index=%d aux_rsc_index=%d tl_bitmap=0x%"PRIx64,
                    wireup_pack_ctx.msg.aux_index, wireup_pack_ctx.msg.tl_index,
                    rsc_index, aux_rsc_index, tl_bitmap);
    }

    packed_len = uct_ep_am_bcopy(ucp_wireup_msg_ep(ep), UCP_AM_ID_WIREUP,
                                 ucp_wireup_msg_pack, &wireup_pack_ctx);
    if (packed_len < 0) {
        status = (ucs_status_t)packed_len;
        if (status != UCS_ERR_NO_RESOURCE) {
            ucs_error("failed to send wireup msg: %s", ucs_status_string(status));
        }
    } else {
        status = UCS_OK;
        if (ep->state & UCP_EP_STATE_STUB_EP) {
            ucs_atomic_add32(&ucp_ep_get_stub_ep(ep)->pending_count, -1);
        }
        ucs_mpool_put(req);
    }

    ucs_free(wireup_pack_ctx.address);
    return status;
}

static ucs_status_t ucp_ep_wireup_send(ucp_ep_h ep, uint8_t type,
                                       ucp_rsc_index_t aux_rsc_index)
{
    ucp_request_t* req;

    req = ucs_mpool_get(&ep->worker->req_mp);
    if (req == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    req->flags                         = UCP_REQUEST_FLAG_RELEASED;
    req->cb.send                       = (ucp_send_callback_t)ucs_empty_function;
    req->send.uct.func                 = ucp_wireup_msg_progress;
    req->send.wireup.type              = type;
    req->send.wireup.aux_rsc_index     = aux_rsc_index;
    if (ep->state & UCP_EP_STATE_STUB_EP) {
        ucs_atomic_add32(&ucp_ep_get_stub_ep(ep)->pending_count, 1);
    }

    ucs_trace("add pending wireup req %p", req);
    ucp_ep_add_pending(ep, ucp_wireup_msg_ep(ep), req, 0);
    return UCS_OK;
}

/*
 * Until the transport is connected, send operations should return NO_RESOURCE.
 * Plant a stub endpoint object which will do it.
 */
ucs_status_t ucp_wireup_create_stub_ep(ucp_ep_h ep)
{
    ucs_status_t status;

    if (ep->state & UCP_EP_STATE_STUB_EP) {
        return UCS_OK;
    }

    status = UCS_CLASS_NEW(ucp_stub_ep_t, &ep->uct_ep, ep);
    if (status != UCS_OK) {
        return status;
    }

    ep->state |= UCP_EP_STATE_STUB_EP;
    ucs_debug("created stub ep %p to %s", ep->uct_ep, ucp_ep_peer_name(ep));
    return UCS_OK;
}

static ucs_status_t ucp_wireup_start_aux(ucp_ep_h ep,
                                         ucp_rsc_index_t dst_pd_index,
                                         ucp_rsc_index_t aux_rsc_index,
                                         const ucp_address_entry_t *aux_addr)
{
    ucp_worker_h worker = ep->worker;
    ucs_status_t status;

    status = ucp_wireup_create_stub_ep(ep);
    if (status != UCS_OK) {
        goto err;
    }

    /*
     * Create auxiliary endpoint which would be used to transfer wireup messages.
     */
    ucs_assert(aux_addr->tl_addr_len > 0);
    status = uct_ep_create_connected(worker->ifaces[aux_rsc_index],
                                     aux_addr->dev_addr, aux_addr->iface_addr,
                                     &ucp_ep_get_stub_ep(ep)->aux_ep);
    if (status != UCS_OK) {
        goto err_destroy_stub_ep;
    }

    ep->state |= UCP_EP_STATE_AUX_EP;
    ucs_debug("created connected aux ep %p to %s using " UCT_TL_RESOURCE_DESC_FMT,
              ucp_ep_get_stub_ep(ep)->aux_ep, ucp_ep_peer_name(ep),
              UCT_TL_RESOURCE_DESC_ARG(&worker->context->tl_rscs[aux_rsc_index].tl_rsc));

    /*
     * Create endpoint for the transport we need to wire-up.
     */
    status = uct_ep_create(worker->ifaces[ep->rsc_index],
                           &ucp_ep_get_stub_ep(ep)->next_ep);
    if (status != UCS_OK) {
        goto err_destroy_aux_ep;
    }

    ep->dst_pd_index = dst_pd_index;

    ep->state |= UCP_EP_STATE_NEXT_EP;
    ucs_debug("created next ep %p to %s using " UCT_TL_RESOURCE_DESC_FMT,
              ucp_ep_get_stub_ep(ep)->next_ep, ucp_ep_peer_name(ep),
              UCT_TL_RESOURCE_DESC_ARG(&worker->context->tl_rscs[ep->rsc_index].tl_rsc));

    /*
     * Send initial connection request for wiring-up the transport
     */
    status = ucp_ep_wireup_send(ep, UCP_WIREUP_MSG_REQUEST, aux_rsc_index);
    if (status != UCS_OK) {
        goto err_destroy_aux_ep;
    }

    return UCS_OK;

err_destroy_aux_ep:
    uct_ep_destroy(ucp_ep_get_stub_ep(ep)->aux_ep);
err_destroy_stub_ep:
    uct_ep_destroy(ep->uct_ep);
err:
    ep->state &= ~(UCP_EP_STATE_STUB_EP|UCP_EP_STATE_AUX_EP|UCP_EP_STATE_NEXT_EP);
    return status;
}

static void ucp_wireup_stop_aux(ucp_ep_h ep)
{
    uct_ep_h uct_eps_to_destroy[3];
    unsigned i, num_eps;

    num_eps = 0;

    ucs_debug("%p: ucp_wireup_stop_aux state=0x%x", ep,
              ep->state);

    if (ep->state & UCP_EP_STATE_NEXT_EP) {
        ucs_assert(ucp_ep_get_stub_ep(ep)->next_ep != NULL);
        uct_eps_to_destroy[num_eps++] = ucp_ep_get_stub_ep(ep)->next_ep;
        ep->state &= ~(UCP_EP_STATE_NEXT_EP|
                       UCP_EP_STATE_NEXT_EP_LOCAL_CONNECTED|
                       UCP_EP_STATE_NEXT_EP_REMOTE_CONNECTED);
        ucp_ep_get_stub_ep(ep)->next_ep = NULL;
    }

    if (ep->state & UCP_EP_STATE_AUX_EP) {
        ucs_assert(ucp_ep_get_stub_ep(ep)->aux_ep != NULL);
        uct_eps_to_destroy[num_eps++] = ucp_ep_get_stub_ep(ep)->aux_ep;
        ucp_ep_get_stub_ep(ep)->aux_ep = NULL;
        ep->state &= ~UCP_EP_STATE_AUX_EP;
    }

    if (ep->state & UCP_EP_STATE_STUB_EP) {
        ucs_assert(ucp_ep_get_stub_ep(ep) != NULL);
        uct_eps_to_destroy[num_eps++] = &ucp_ep_get_stub_ep(ep)->super;
        ep->uct_ep = NULL;
        ep->state &= ~UCP_EP_STATE_STUB_EP;
    }

    for (i = 0; i < num_eps; ++i) {
        ucp_ep_destroy_uct_ep_safe(ep, uct_eps_to_destroy[i]);
    }
}

static void ucp_wireup_process_request(ucp_worker_h worker, ucp_wireup_msg_t *msg,
                                       uint64_t uuid, const char *peer_name,
                                       unsigned address_count,
                                       const ucp_address_entry_t *address_list)
{
    ucp_ep_h ep = ucp_worker_ep_find(worker, uuid);
    const ucp_address_entry_t *tl_addr, *aux_addr;
    uct_iface_attr_t *iface_attr;
    ucp_rsc_index_t aux_rsc_index;
    unsigned addr_index;
    ucs_status_t status;
    uct_ep_h uct_ep;

    ucs_assert(msg->tl_index != (uint8_t)-1);
    tl_addr = &address_list[msg->tl_index];

    if (ep == NULL) {
        status = ucp_ep_new(worker, uuid, peer_name, "remote-request", &ep);
        if (status != UCS_OK) {
            return;
        }

     } else if (ep->state & UCP_EP_STATE_NEXT_EP_LOCAL_CONNECTED) {
        /* TODO possibly switch to different transport */
        ucs_assertv(ep->dst_pd_index == tl_addr->pd_index,
                    "ep->dst_pd_index=%d tl_addr->pd_index=%d", ep->dst_pd_index,
                    tl_addr->pd_index);
        ucs_assert(ucp_ep_pd_index(ep) == msg->dst_pd_index);
        ucs_trace("ignoring connection request - already connected");
        return;
    }

    status = ucp_select_transport(worker, peer_name, tl_addr, 1, msg->dst_pd_index,
                                  &ep->rsc_index, &addr_index,
                                  ucp_runtime_score_func, "runtime-on-demand");
    if (status != UCS_OK) {
        return;
    }

    ucs_assert(addr_index == 0);

    ep->dst_pd_index = tl_addr->pd_index;

    ucs_debug("using " UCT_TL_RESOURCE_DESC_FMT,
              UCT_TL_RESOURCE_DESC_ARG(&worker->context->tl_rscs[ep->rsc_index].tl_rsc));

    iface_attr = &worker->iface_attrs[ep->rsc_index];

    if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {

        status = uct_ep_create_connected(worker->ifaces[ep->rsc_index],
                                         tl_addr->dev_addr, tl_addr->iface_addr,
                                         &uct_ep);
        if (status != UCS_OK) {
            ucs_debug("failed to create ep");
            return;
        }

        if (ep->state & UCP_EP_STATE_STUB_EP) {
             ucp_ep_get_stub_ep(ep)->next_ep = uct_ep;
             ep->state |= UCP_EP_STATE_NEXT_EP;
             ep->state |= UCP_EP_STATE_NEXT_EP_LOCAL_CONNECTED;
             ep->state |= UCP_EP_STATE_NEXT_EP_REMOTE_CONNECTED;
        } else {
             ep->uct_ep = uct_ep;
             ep->state |= UCP_EP_STATE_READY_TO_SEND;
             ep->state |= UCP_EP_STATE_READY_TO_RECEIVE;
        }

        /* Send a reply, which also includes the address of next_ep */
        status = ucp_ep_wireup_send(ep, UCP_WIREUP_MSG_ACK, UCP_NULL_RESOURCE);
        if (status != UCS_OK) {
            return;
        }

        ucs_memory_cpu_fence();
        ep->state |= UCP_EP_STATE_WIREUP_ACK_SENT;

    } else if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {

        /* Create auxiliary transport and put the real transport in next_ep */
        if (((msg->aux_index != (uint8_t)-1)) && (!(ep->state & UCP_EP_STATE_AUX_EP))) {

            ucs_assert(!(ep->state & UCP_EP_STATE_NEXT_EP));

            aux_addr = &address_list[msg->aux_index];
            status = ucp_select_transport(worker, ucp_ep_peer_name(ep),
                                          aux_addr, 1, UCP_NULL_RESOURCE,
                                          &aux_rsc_index, &addr_index,
                                          ucp_aux_score_func, "aux-on-demand");
            if (status != UCS_OK) {
                ucs_error("No suitable auxiliary transport found");
                return;
            }

            status = ucp_wireup_start_aux(ep, tl_addr->pd_index, aux_rsc_index,
                                          aux_addr);
            if (status != UCS_OK) {
                return;
            }
        }

        ucs_assert(ep->state & UCP_EP_STATE_NEXT_EP);

        /* Connect next_ep to the address passed in the wireup message */
        if (!(ep->state & UCP_EP_STATE_NEXT_EP_LOCAL_CONNECTED)) {
            ucs_assert(ep->state & UCP_EP_STATE_NEXT_EP);
            status = uct_ep_connect_to_ep(ucp_ep_get_stub_ep(ep)->next_ep,
                                          tl_addr->dev_addr, tl_addr->ep_addr);
            if (status != UCS_OK) {
                /* TODO send reject? */
                ucs_debug("failed to connect");
                return;
            }

            ucs_debug("connected next ep %p", ucp_ep_get_stub_ep(ep)->next_ep);
            ep->state |= UCP_EP_STATE_NEXT_EP_LOCAL_CONNECTED;
        }

        /* Send a reply, which also includes the address of next_ep */
        status = ucp_ep_wireup_send(ep, UCP_WIREUP_MSG_REPLY, -1);
        if (status != UCS_OK) {
            return;
        }

        ucs_memory_cpu_fence();
        ep->state |= UCP_EP_STATE_WIREUP_REPLY_SENT;
    }
}

static void ucp_wireup_process_reply(ucp_worker_h worker, ucp_wireup_msg_t *msg,
                                     uint64_t uuid, unsigned address_count,
                                     const ucp_address_entry_t *address_list)
{
    ucp_ep_h ep = ucp_worker_ep_find(worker, uuid);
    const ucp_address_entry_t *tl_addr;
    ucs_status_t status;

    if (ep == NULL) {
        ucs_debug("ignoring connection reply - not exists");
        return;
    }

    if (ep->state & UCP_EP_STATE_READY_TO_SEND) {
        ucs_debug("ignoring conn_rep - already connected");
        return;
    }

    if (ep->state & UCP_EP_STATE_NEXT_EP_REMOTE_CONNECTED) {
        ucs_debug("ignoring conn_rep - remote already connected");
        return;
    }

    ucs_assert_always(ep->state & UCP_EP_STATE_NEXT_EP);

    /*
     * If we got a reply, it means the remote side got our address and connected
     * to us.
     */
    ep->state |= UCP_EP_STATE_NEXT_EP_REMOTE_CONNECTED;

    /* If we have not connected yet, do it now */
    if (!(ep->state & UCP_EP_STATE_NEXT_EP_LOCAL_CONNECTED)) {
        ucs_assert(ep->state & UCP_EP_STATE_NEXT_EP);

        tl_addr = &address_list[msg->tl_index];
        status = uct_ep_connect_to_ep(ucp_ep_get_stub_ep(ep)->next_ep,
                                      tl_addr->dev_addr, tl_addr->ep_addr);
        if (status != UCS_OK) {
            ucs_error("failed to connect");
            return;
        }

        /* The remote side must be connected with same PD we asked for */
        ucs_assert(ep->dst_pd_index == tl_addr->pd_index);

        /* The remote side should ask to be connected to our PD */
        ucs_assert(ucp_ep_pd_index(ep) == msg->dst_pd_index);

        ucs_debug("connected next ep %p pd %d", ucp_ep_get_stub_ep(ep)->next_ep,
                  ep->dst_pd_index);

        ep->state |= UCP_EP_STATE_NEXT_EP_LOCAL_CONNECTED;
    }

    /* If we already sent a reply to the remote side, no need to send an ACK. */
    if (ep->state & (UCP_EP_STATE_WIREUP_REPLY_SENT|UCP_EP_STATE_WIREUP_ACK_SENT)) {
        ucs_debug("ep %p not sending wireup_ack - state is 0x%x", ep, ep->state);
        return;
    }

    /*
     * Send ACK to let remote side know it can start sending.
     *
     * We can use the new ep even from async thread, because main thread will not
     * started using it before ACK_SENT is turned on.
     */
    status = ucp_ep_wireup_send(ep, UCP_WIREUP_MSG_ACK, UCP_NULL_RESOURCE);
    if (status != UCS_OK) {
        return;
    }

    ucs_memory_cpu_fence();
    ep->state |= UCP_EP_STATE_WIREUP_ACK_SENT;
}

static void ucp_wireup_process_ack(ucp_worker_h worker, uint64_t uuid)
{
    ucp_ep_h ep = ucp_worker_ep_find(worker, uuid);

    if (ep == NULL) {
        ucs_debug("ignoring connection ack - ep not exists");
        return;
    }

    if (ep->state & UCP_EP_STATE_NEXT_EP_REMOTE_CONNECTED) {
        ucs_debug("ignoring connection ack - remote already connected");
        return;
    }

    /*
     * If we got CONN_ACK, it means remote side got our reply, and also
     * connected to us.
     */
    if (ep->state & UCP_EP_STATE_READY_TO_SEND) {
        ep->state |= UCP_EP_STATE_READY_TO_RECEIVE;
    } else {
        ep->state |= UCP_EP_STATE_NEXT_EP_REMOTE_CONNECTED;
    }
}

static ucs_status_t ucp_wireup_msg_handler(void *arg, void *data,
                                           size_t length, void *desc)
{
    ucp_worker_h worker   = arg;
    ucp_wireup_msg_t *msg = data;
    char peer_name[UCP_WORKER_NAME_MAX];
    ucp_address_entry_t *address_list;
    unsigned address_count;
    ucs_status_t status;
    uint64_t uuid;

    UCS_ASYNC_BLOCK(&worker->async);

    status = ucp_address_unpack(msg + 1, &uuid, peer_name, UCP_WORKER_NAME_MAX,
                                &address_count, &address_list);
    if (status != UCS_OK) {
        ucs_error("failed to unpack address: %s", ucs_status_string(status));
        goto out;
    }

    if (msg->type == UCP_WIREUP_MSG_ACK) {
        ucs_assert(address_count == 0);
        ucp_wireup_process_ack(worker, uuid);
    } else if (msg->type == UCP_WIREUP_MSG_REQUEST) {
        ucp_wireup_process_request(worker, msg, uuid, peer_name, address_count,
                                   address_list);
    } else if (msg->type == UCP_WIREUP_MSG_REPLY) {
        ucp_wireup_process_reply(worker, msg, uuid, address_count, address_list);
    } else {
        ucs_bug("invalid wireup message");
    }

    ucs_free(address_list);

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return UCS_OK;
}

ucs_status_t ucp_wireup_start(ucp_ep_h ep, ucp_address_entry_t *address_list,
                              unsigned address_count)
{
    ucp_worker_h worker = ep->worker;
    uct_iface_attr_t *iface_attr;
    unsigned addr_index, aux_addr_index;
    ucp_rsc_index_t aux_rsc_index;
    ucs_status_t status;

    UCS_ASYNC_BLOCK(&worker->async);

    /*
     * Select best transport for runtime
     */
    status = ucp_select_transport(worker, ucp_ep_peer_name(ep), address_list,
                                  address_count, UCP_NULL_RESOURCE,
                                  &ep->rsc_index, &addr_index,
                                  ucp_runtime_score_func, "runtime");
    if (status != UCS_OK) {
        goto out;
    }

    iface_attr = &worker->iface_attrs[ep->rsc_index];

    /*
     * If the selected transport can be connected directly, do it.
     */
    if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
        ucs_assert(address_list[addr_index].tl_addr_len > 0);
        status = uct_ep_create_connected(worker->ifaces[ep->rsc_index],
                                         address_list[addr_index].dev_addr,
                                         address_list[addr_index].iface_addr,
                                         &ep->uct_ep);
        if (status != UCS_OK) {
            ucs_debug("failed to create ep");
            goto out;
        }

        ep->dst_pd_index = address_list[addr_index].pd_index;

        ucs_debug("created connected ep %p to %s pd %d using " UCT_TL_RESOURCE_DESC_FMT,
                  ep->uct_ep, ucp_ep_peer_name(ep), ep->dst_pd_index,
                  UCT_TL_RESOURCE_DESC_ARG(&worker->context->tl_rscs[ep->rsc_index].tl_rsc));
        ucp_wireup_ep_ready_to_send(ep);
        goto out;
    } else if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP)) {
        status = UCS_ERR_UNREACHABLE;
        goto out;
    }

    /*
     * If we cannot connect the selected transport directly, select another
     * transport to be an auxiliary.
     */
    status = ucp_select_transport(worker, ucp_ep_peer_name(ep), address_list,
                                  address_count, UCP_NULL_RESOURCE,
                                  &aux_rsc_index, &aux_addr_index,
                                  ucp_aux_score_func, "auxiliary");
    if (status != UCS_OK) {
        goto out;
    }

    /*
     * Start connection establishment protocol on the auxiliary address.
     */
    status = ucp_wireup_start_aux(ep, address_list[addr_index].pd_index,
                                  aux_rsc_index, &address_list[aux_addr_index]);
    if (status != UCS_OK) {
        goto out;
    }

    status = UCS_OK;

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return status;
}

void ucp_wireup_stop(ucp_ep_h ep)
{
    ucp_worker_h worker = ep->worker;

    ucs_trace_func("ep=%p", ep);

    UCS_ASYNC_BLOCK(&worker->async);
    ucp_wireup_stop_aux(ep);
    UCS_ASYNC_UNBLOCK(&worker->async);
}

ucs_status_t ucp_wireup_connect_remote(ucp_ep_h ep)
{
    ucs_status_t status;

    ucs_assert(!(ep->state & UCP_EP_STATE_READY_TO_RECEIVE));
    status = ucp_ep_wireup_send(ep, UCP_WIREUP_MSG_REQUEST, -1);
    return status;
}

UCP_DEFINE_AM(-1, UCP_AM_ID_WIREUP, ucp_wireup_msg_handler, 
              ucp_wireup_msg_dump, UCT_AM_CB_FLAG_ASYNC);
