/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "wireup.h"
#include "stub_ep.h"

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/dt/dt_contig.h>
#include <ucp/tag/eager.h>
#include <ucs/arch/atomic.h>
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

static ucp_stub_ep_t * ucp_ep_get_stub_ep(ucp_ep_h ep)
{
    ucs_assert(ep->state & UCP_EP_STATE_STUB_EP);
    return ucs_derived_of(ep->uct_ep, ucp_stub_ep_t);
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

/*
 * UCP address is a serialization of multiple interface addresses.
 * It is built from records which look like this:
 *
 * [ entity UUID ]
 *
 * +------------+------+----------------------+------------------+
 * | tl_name[n] | '\0' | address_len (1 byte) | uct_iface_addr_t |
 * +------------+------+----------------------+------------------+
 *
 * ...
 *
 * The last record starts with '\0' (as if the tl_name is empty string)
 */


static int ucp_address_iter_next(void **iter, uct_iface_addr_t **iface_addr_p,
                                 char *tl_name, ucp_rsc_index_t *pd_index,
                                 ucp_rsc_index_t *rsc_index)
{
    char *ptr = *iter;
    uint8_t iface_addr_len;
    uint8_t tl_name_len;

    iface_addr_len = *(uint8_t*)(ptr++);
    if (iface_addr_len == 0) {
        return 0;
    }

    *iface_addr_p  = (uct_iface_addr_t*)ptr;
    ptr += iface_addr_len;

    tl_name_len = *(uint8_t*)(ptr++);
    ucs_assert_always(tl_name_len < UCT_TL_NAME_MAX);
    memcpy(tl_name, ptr, tl_name_len);
    tl_name[tl_name_len] = '\0';
    ptr += tl_name_len;

    *pd_index  = *((ucp_rsc_index_t*)ptr++);
    *rsc_index = *((ucp_rsc_index_t*)ptr++);

    *iter = ptr;
    return 1;
}

static inline void * ucp_wireup_msg_addr(ucp_wireup_msg_t *msg)
{
    void *msg_data = msg + 1;
    return msg_data + msg->peer_name_len + msg->tl_name_len;
}

static const void * ucp_wireup_msg_get_addr(const ucp_wireup_msg_t *msg)
{
    const void *msg_data = msg + 1;
    return msg_data + msg->peer_name_len + msg->tl_name_len;
}

static inline char* ucp_wireup_msg_tl_name(ucp_wireup_msg_t *msg)
{
    void *msg_data = msg + 1;
    return msg_data + msg->peer_name_len;
}

static inline char* ucp_wireup_msg_peer_name(ucp_wireup_msg_t *msg)
{
    return (char *)(msg + 1);
}

static void ucp_wireup_msg_get_peer_name(const ucp_wireup_msg_t *msg,
                                         char *peer_name, size_t max)
{
    size_t length = ucs_min(max - 1, msg->peer_name_len);
    memcpy(peer_name, msg + 1, length);
    peer_name[length] = '\0';
}

static void ucp_wireup_msg_get_tl_name(const ucp_wireup_msg_t *msg,
                                       char *tl_name, size_t max)
{
    size_t length = ucs_min(max - 1, msg->tl_name_len);
    memcpy(tl_name, (const char*)(msg + 1) + msg->peer_name_len, length);
    tl_name[length] = '\0';
}

static uct_iface_addr_t * ucp_wireup_msg_get_aux_addr(ucp_wireup_msg_t *msg)
{
    void *msg_data = msg + 1;
    return msg_data + msg->peer_name_len + msg->tl_name_len + msg->addr_len;
}

static double ucp_aux_score_func(ucp_worker_h worker, uct_iface_attr_t *iface_attr)
{
    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_BCOPY) || /* Need to use it for wireup messages */
        !(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) || /* Should connect immediately */
        !(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_CB_ASYNC) || /* Should progress asynchronously */
        !(iface_attr->cap.flags & UCT_IFACE_FLAG_PENDING) /* Should support pending operations */ )
    {
        return 0.0;
    }

    return (1e-3 / iface_attr->latency) +
           (1e3 * ucs_max(iface_attr->cap.am.max_bcopy, iface_attr->cap.am.max_short));
}

static double ucp_runtime_score_func(ucp_worker_h worker, uct_iface_attr_t *iface_attr)
{
    ucp_context_t *context = worker->context;
    uint64_t flags;

    flags = 0;

    if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
        flags |= UCT_IFACE_FLAG_AM_BCOPY;
    }

    if (context->config.features & UCP_FEATURE_TAG) {
        flags |= UCT_IFACE_FLAG_AM_SHORT | UCT_IFACE_FLAG_PENDING | UCT_IFACE_FLAG_AM_CB_SYNC;
    }

    if (context->config.features & UCP_FEATURE_RMA) {
        /* TODO remove this requirement once we have RMA emulation */
        flags |= UCT_IFACE_FLAG_PUT_SHORT | UCT_IFACE_FLAG_PUT_BCOPY |
                 UCT_IFACE_FLAG_GET_BCOPY;
    }

    if (context->config.features & UCP_FEATURE_AMO32) {
        /* TODO remove this requirement once we have SW atomics */
        flags |= UCT_IFACE_FLAG_ATOMIC_ADD32 | UCT_IFACE_FLAG_ATOMIC_FADD32 |
                 UCT_IFACE_FLAG_ATOMIC_SWAP32 | UCT_IFACE_FLAG_ATOMIC_CSWAP32;
    }

    if (context->config.features & UCP_FEATURE_AMO64) {
        /* TODO remove this requirement once we have SW atomics */
        flags |= UCT_IFACE_FLAG_ATOMIC_ADD64 | UCT_IFACE_FLAG_ATOMIC_FADD64 |
                 UCT_IFACE_FLAG_ATOMIC_SWAP64 | UCT_IFACE_FLAG_ATOMIC_CSWAP64;
    }

    if (!ucs_test_all_flags(iface_attr->cap.flags, flags)) {
        return 0.0;
    }

    return 1e-3 / (iface_attr->latency + (iface_attr->overhead * 2));
}

/*
 * Select which local transport to use for a particular destination address.
 */
static ucs_status_t
ucp_select_local_transport(ucp_worker_h worker, const uct_iface_addr_t *remote_addr,
                           const char *tl_name, ucp_wireup_score_function_t score_func,
                           ucp_rsc_index_t dst_rsc_index,
                           ucp_rsc_index_t *src_rsc_index_p,
                           double *score_p, const char *title)
{
    ucp_context_h context = worker->context;
    ucp_rsc_index_t src_rsc_index;
    double score, best_score;
    uct_tl_resource_desc_t *resource;
    uct_iface_h iface;
    int reachable;

    best_score       = 0.0;
    *src_rsc_index_p = -1;

    for (src_rsc_index = 0; src_rsc_index < context->num_tls; ++src_rsc_index) {
        resource   = &context->tl_rscs[src_rsc_index].tl_rsc;
        iface      = worker->ifaces[src_rsc_index];

        /* Must be reachable address, on same transport */
        reachable = !strcmp(tl_name, resource->tl_name) &&
                     uct_iface_is_reachable(iface, remote_addr);
        ucs_trace("'%s' is %sreachable from "UCT_TL_RESOURCE_DESC_FMT,
                  tl_name, reachable ? "" : "not ",
                                  UCT_TL_RESOURCE_DESC_ARG(resource));
        if (!reachable) {
            continue;
        }

        score = score_func(worker, &worker->iface_attrs[src_rsc_index]);
        ucs_trace("%s " UCT_TL_RESOURCE_DESC_FMT " [%d->%d] score %.2f",
                  title, UCT_TL_RESOURCE_DESC_ARG(resource), src_rsc_index,
                  dst_rsc_index, score);
        if (score > best_score) {
            best_score       = score;
            *src_rsc_index_p = src_rsc_index;
        }
    }

    if (*src_rsc_index_p == (ucp_rsc_index_t)-1) {
        return UCS_ERR_UNREACHABLE;
    }

    *score_p = best_score;
    return UCS_OK;
}

static ucs_status_t ucp_select_transport(ucp_worker_h worker, ucp_address_t *address,
                                         ucp_wireup_score_function_t score_func,
                                         ucp_rsc_index_t *src_rsc_index_p,
                                         ucp_rsc_index_t *dst_rsc_index_p,
                                         ucp_rsc_index_t *dst_pd_index_p,
                                         uct_iface_addr_t **addr_p,
                                         uint64_t *reachable_pds,
                                         const char *title)
{
    ucp_context_h context = worker->context;
    ucp_rsc_index_t dst_rsc_index, src_rsc_index;
    ucp_rsc_index_t pd_index;
    uct_iface_addr_t *addr, *best_addr;
    double score, best_score;
    char tl_name[UCT_TL_NAME_MAX];
    ucs_status_t status;
    void *iter;

    best_addr        = NULL;
    best_score       = 1e-9;
    *src_rsc_index_p = -1;
    *dst_rsc_index_p = -1;
    *dst_pd_index_p  = -1;
    *reachable_pds   = 0;

    /*
     * Find the best combination of local resource and reachable remote address.
     */
    iter = ucp_address_iter_start(address);
    while (ucp_address_iter_next(&iter, &addr, tl_name, &pd_index, &dst_rsc_index)) {
        status = ucp_select_local_transport(worker, addr, tl_name, score_func,
                                            dst_rsc_index, &src_rsc_index,
                                            &score, title);
        if ((status == UCS_OK) && (score > best_score)) {
            ucs_assert(addr != NULL);
            best_score       = score;
            best_addr        = addr;
            *dst_rsc_index_p = dst_rsc_index;
            *src_rsc_index_p = src_rsc_index;
            *dst_pd_index_p  = pd_index;
        }
    }

    if (best_addr == NULL) {
        return UCS_ERR_UNREACHABLE;
    }

    ucs_debug("selected for %s: " UCT_TL_RESOURCE_DESC_FMT " [%d->%d] pd %d", title,
              UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[*src_rsc_index_p].tl_rsc),
              *src_rsc_index_p, *dst_rsc_index_p, *dst_pd_index_p);
    *addr_p = best_addr;
    return UCS_OK;
}

static void ucp_wireup_msg_dump(ucp_worker_h worker, uct_am_trace_type_t type,
                                uint8_t id, const void *data, size_t length,
                                char *buffer, size_t max)
{
    ucp_context_h context = worker->context;
    const ucp_wireup_msg_t *msg = data;
    char peer_name[UCP_PEER_NAME_MAX + 1];
    char tl_name[UCT_TL_NAME_MAX + 1];
    uct_tl_resource_desc_t *dst_rsc;
    uct_tl_resource_desc_t dummy_rsc = {"", "", UCT_DEVICE_TYPE_LAST};

    #define UCP_WIREUP_MSG_FLAGS_FMT  "%c%c%c%c%c"
    #define UCP_WIREUP_MSG_FLAGS_ARG(_flags) \
        ((_flags) & UCP_WIREUP_FLAG_REQUSET ) ? 'q' : '-', \
        ((_flags) & UCP_WIREUP_FLAG_REPLY   ) ? 'p' : '-', \
        ((_flags) & UCP_WIREUP_FLAG_ACK     ) ? 'a' : '-', \
        ((_flags) & UCP_WIREUP_FLAG_ADDR    ) ? 't' : '-', \
        ((_flags) & UCP_WIREUP_FLAG_AUX_ADDR) ? 'x' : '-'

    ucp_wireup_msg_get_peer_name(msg, peer_name, sizeof(peer_name));
    ucp_wireup_msg_get_tl_name(msg, tl_name, sizeof(tl_name));

    switch (type) {
    case UCT_AM_TRACE_TYPE_SEND:
        snprintf(buffer, max, "WIREUP "UCP_WIREUP_MSG_FLAGS_FMT" "
                 "[%s uuid %"PRIx64" "UCT_TL_RESOURCE_DESC_FMT",%s->#%d tl '%s']",
                 UCP_WIREUP_MSG_FLAGS_ARG(msg->flags), peer_name, msg->src_uuid,
                 UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[msg->src_rsc_index].tl_rsc),
                 context->pd_rscs[msg->src_pd_index].pd_name,
                 msg->dst_rsc_index, tl_name);
        break;
    case UCT_AM_TRACE_TYPE_RECV:
        if (msg->dst_rsc_index != (ucp_rsc_index_t)-1) {
            dst_rsc = &context->tl_rscs[msg->dst_rsc_index].tl_rsc;
        } else {
            dst_rsc = &dummy_rsc;
        }
        snprintf(buffer, max, "WIREUP "UCP_WIREUP_MSG_FLAGS_FMT" "
                 "[%s uuid %"PRIx64" #%d,%d->"UCT_TL_RESOURCE_DESC_FMT" tl '%s']",
                 UCP_WIREUP_MSG_FLAGS_ARG(msg->flags), peer_name, msg->src_uuid,
                 msg->src_rsc_index, msg->src_pd_index,
                 UCT_TL_RESOURCE_DESC_ARG(dst_rsc), tl_name);
        break;
    default:
        break;
    }
}

static ucs_status_t ucp_wireup_send_am(ucp_ep_h ep, uct_ep_h uct_ep, uint16_t flags,
                                       ucp_rsc_index_t dst_rsc_index,
                                       ucp_rsc_index_t dst_aux_rsc_index)
{
    ucp_rsc_index_t rsc_index    = ep->rsc_index;
    ucp_worker_h worker          = ep->worker;
    ucp_context_h context        = worker->context;
    uct_iface_attr_t *iface_attr = &worker->iface_attrs[rsc_index];
    uct_tl_resource_desc_t *rsc  = &context->tl_rscs[rsc_index].tl_rsc;
    char worker_name[UCP_PEER_NAME_MAX];
    uct_iface_attr_t aux_iface_attr;
    size_t addr_len = 0, aux_addr_len = 0, total_len;
    ucp_memcpy_pack_context_t pack_ctx;
    ucp_wireup_msg_t *msg;
    ucs_status_t status;
    ssize_t packed_len;

    /* Get runtime address length */
    if (flags & UCP_WIREUP_FLAG_ADDR) {
        if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
            addr_len = iface_attr->iface_addr_len;
        } else if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
            addr_len = iface_attr->ep_addr_len;
        }
    }

    /* Get auxiliary address length */
    if (flags & UCP_WIREUP_FLAG_AUX_ADDR) {
        ucs_assert(ep->state & UCP_EP_STATE_AUX_EP);
        status = uct_iface_query(ucp_ep_get_stub_ep(ep)->aux_ep->iface,
                                 &aux_iface_attr);
        if (status != UCS_OK) {
            goto err;
        }

        ucs_assert(aux_iface_attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE);
        aux_addr_len = aux_iface_attr.iface_addr_len;
    }

    ucp_worker_get_name(worker, worker_name, UCP_PEER_NAME_MAX);

    /*
     * Allocate buffer for active message.
     * TODO use custom pack callback to avoid this allocation and memcpy
     */
    total_len = sizeof(*msg) +
                strlen(worker_name) +
                strlen(rsc->tl_name) +
                addr_len + aux_addr_len;
    msg = ucs_malloc(total_len, "conn_req");
    if (msg == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* Fill message header */
    msg->src_uuid      = worker->uuid;
    msg->src_pd_index  = context->tl_rscs[rsc_index].pd_index;
    msg->src_rsc_index = rsc_index;
    msg->dst_rsc_index = dst_rsc_index;
    msg->dst_aux_index = dst_aux_rsc_index;
    msg->flags         = flags;
    msg->addr_len      = addr_len;

    /* Copy peer name (debug mode only) */
    UCS_STATIC_ASSERT(UCP_PEER_NAME_MAX < UINT8_MAX);
    msg->peer_name_len = strlen(worker_name);
    memcpy(ucp_wireup_msg_peer_name(msg), worker_name, msg->peer_name_len);

    /* Copy transport name */
    msg->tl_name_len = strlen(rsc->tl_name);
    UCS_STATIC_ASSERT(UCT_TL_NAME_MAX < UINT8_MAX);
    memcpy(ucp_wireup_msg_tl_name(msg), rsc->tl_name, msg->tl_name_len);

    /* Fill runtime address */
    if (flags & UCP_WIREUP_FLAG_ADDR) {
        if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
            ucs_assert(ep->state & UCP_EP_STATE_READY_TO_SEND);
            status = uct_iface_get_address(worker->ifaces[rsc_index],
                                           ucp_wireup_msg_addr(msg));
        } else if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
            ucs_assert(ep->state & UCP_EP_STATE_NEXT_EP);
            status = uct_ep_get_address(ucp_ep_get_stub_ep(ep)->next_ep,
                                        ucp_wireup_msg_addr(msg));
        } else {
            status = UCS_ERR_UNREACHABLE;
        }
        if (status != UCS_OK) {
            goto err_free;
        }
    }

    /* Fill auxiliary address */
    if (flags & UCP_WIREUP_FLAG_AUX_ADDR) {
        ucs_assert(ep->state & UCP_EP_STATE_AUX_EP);
        status = uct_iface_get_address(ucp_ep_get_stub_ep(ep)->aux_ep->iface,
                                       ucp_wireup_msg_get_aux_addr(msg));
        if (status != UCS_OK) {
            goto err_free;
        }
    }

    /* Send active message */
    pack_ctx.src    = msg;
    pack_ctx.length = total_len;
    packed_len = uct_ep_am_bcopy(uct_ep, UCP_AM_ID_WIREUP, ucp_memcpy_pack, &pack_ctx);
    if (packed_len < 0) {
        status = (ucs_status_t)packed_len;
        if (status != UCS_ERR_NO_RESOURCE) {
            ucs_error("failed to send conn msg: %s", ucs_status_string(status));
        }
        goto err_free;
    }

    ucs_free(msg);
    return UCS_OK;

err_free:
    ucs_free(msg);
err:
    return status;
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

static ucs_status_t ucp_ep_wireup_op_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_h ep = req->send.ep;
    ucs_status_t status;

    status = ucp_wireup_send_am(ep, ucp_wireup_msg_ep(ep), req->send.wireup.flags,
                                req->send.wireup.dst_rsc_index,
                                req->send.wireup.dst_aux_rsc_index);
    if (status == UCS_OK) {
        if (ep->state & UCP_EP_STATE_STUB_EP) {
            ucs_atomic_add32(&ucp_ep_get_stub_ep(ep)->pending_count, -1);
        }
        ucs_mpool_put(req);
    }
    return status;
}

static ucs_status_t ucp_ep_wireup_send(ucp_ep_h ep, uint16_t flags,
                                       ucp_rsc_index_t dst_rsc_index,
                                       ucp_rsc_index_t dst_aux_rsc_index,
                                       int allow_progress)
{
    uct_ep_h uct_ep = ucp_wireup_msg_ep(ep);
    ucp_request_t* req;
    ucs_status_t status;

    status = ucp_wireup_send_am(ep, uct_ep, flags, dst_rsc_index,
                                   dst_aux_rsc_index);
    if (status != UCS_ERR_NO_RESOURCE) {
        return status;
    }

    req = ucs_mpool_get(&ep->worker->req_mp);
    if (req == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    req->flags                         = UCP_REQUEST_FLAG_RELEASED;
    req->cb.send                       = (ucp_send_callback_t)ucs_empty_function;
    req->send.uct.func                 = ucp_ep_wireup_op_progress;
    req->send.wireup.flags             = flags;
    req->send.wireup.dst_rsc_index     = dst_rsc_index;
    req->send.wireup.dst_aux_rsc_index = dst_aux_rsc_index;
    if (ep->state & UCP_EP_STATE_STUB_EP) {
        ucs_atomic_add32(&ucp_ep_get_stub_ep(ep)->pending_count, 1);
    }
    ucp_ep_add_pending(ep, uct_ep, req, allow_progress);
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

static ucs_status_t ucp_wireup_start_aux(ucp_ep_h ep, uct_iface_addr_t *aux_addr,
                                         ucp_rsc_index_t aux_rsc_index)
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
    status = uct_ep_create_connected(worker->ifaces[aux_rsc_index],
                                     aux_addr, &ucp_ep_get_stub_ep(ep)->aux_ep);
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

    ep->state |= UCP_EP_STATE_NEXT_EP;
    ucs_debug("created next ep %p to %s using " UCT_TL_RESOURCE_DESC_FMT,
              ucp_ep_get_stub_ep(ep)->next_ep, ucp_ep_peer_name(ep),
              UCT_TL_RESOURCE_DESC_ARG(&worker->context->tl_rscs[ep->rsc_index].tl_rsc));

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

static void ucp_wireup_process_request(ucp_worker_h worker, ucp_ep_h ep,
                                       ucp_wireup_msg_t *msg)
{
    uct_iface_attr_t *iface_attr;
    ucs_status_t status;
    char peer_name[UCP_PEER_NAME_MAX + 1];
    char tl_name[UCT_TL_NAME_MAX + 1];
    double score;

    if (ep == NULL) {
        ucp_wireup_msg_get_peer_name(msg, peer_name, sizeof(peer_name));
        status = ucp_ep_new(worker, msg->src_uuid, peer_name, "remote-request", &ep);
        if (status != UCS_OK) {
            return;
        }
    }

    if (!(ep->state & UCP_EP_STATE_NEXT_EP_LOCAL_CONNECTED)) {
        ep->dst_pd_index = msg->src_pd_index;

        if (msg->dst_rsc_index == (ucp_rsc_index_t)-1) {
            /* Remote peer did not specify which device to use */
            ucp_wireup_msg_get_tl_name(msg, tl_name, sizeof(tl_name));
            status = ucp_select_local_transport(worker, ucp_wireup_msg_addr(msg),
                                                tl_name, ucp_runtime_score_func,
                                                msg->src_rsc_index, &ep->rsc_index,
                                                &score, "runtime-on-demand");
            if (status != UCS_OK) {
                ucs_error("No suitable transport found");
                return;
            }

            ucs_debug("using " UCT_TL_RESOURCE_DESC_FMT,
                      UCT_TL_RESOURCE_DESC_ARG(&worker->context->tl_rscs[ep->rsc_index].tl_rsc));
        } else {
            ep->rsc_index    = msg->dst_rsc_index;
        }
    }

    if (ep->state & UCP_EP_STATE_READY_TO_SEND) {
         ucs_trace("ignoring connection request - already connected");
         return;
    }

    if ((msg->dst_rsc_index != (ucp_rsc_index_t)-1) && (msg->dst_rsc_index != ep->rsc_index)) {
        ucs_error("got connection request on a different resource (got: %d, expected: %d)",
                  msg->dst_rsc_index, ep->rsc_index);
        /* TODO send reject, and use different transport */
        return;
    }

    iface_attr = &worker->iface_attrs[ep->rsc_index];

    if ((msg->flags & UCP_WIREUP_FLAG_ADDR) &&
        (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE))
    {
        if (ep->state & UCP_EP_STATE_STUB_EP) {
             status = uct_ep_create_connected(worker->ifaces[ep->rsc_index],
                                              ucp_wireup_msg_addr(msg),
                                              &ucp_ep_get_stub_ep(ep)->next_ep);
             if (status != UCS_OK) {
                 return;
             }

             ep->state |= UCP_EP_STATE_NEXT_EP;
             ep->state |= UCP_EP_STATE_NEXT_EP_LOCAL_CONNECTED;
             ep->state |= UCP_EP_STATE_NEXT_EP_REMOTE_CONNECTED;

        } else {
            status = uct_ep_create_connected(worker->ifaces[ep->rsc_index],
                                             ucp_wireup_msg_addr(msg),
                                             &ep->uct_ep);
            if (status != UCS_OK) {
                ucs_debug("failed to create ep");
                return;
            }

            ep->state |= UCP_EP_STATE_READY_TO_SEND | UCP_EP_STATE_READY_TO_RECEIVE;
        }

        /* Send a reply, which also includes the address of next_ep */
        status = ucp_ep_wireup_send(ep, UCP_WIREUP_FLAG_ACK,
                                    msg->src_rsc_index, -1, 0);
        if (status != UCS_OK) {
            return;
        }

        ucs_memory_cpu_fence();
        ep->state |= UCP_EP_STATE_WIREUP_ACK_SENT;

    } else if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {

        if (msg->flags & UCP_WIREUP_FLAG_AUX_ADDR) {
            /* Create auxiliary transport and put the real transport in next_ep */
            if (!(ep->state & UCP_EP_STATE_AUX_EP)) {
                ucs_assert(!(ep->state & UCP_EP_STATE_NEXT_EP));
                status = ucp_wireup_start_aux(ep, ucp_wireup_msg_get_aux_addr(msg),
                                              msg->dst_aux_index);
                if (status != UCS_OK) {
                    return;
                }
            }
        }

        ucs_assert(ep->state & UCP_EP_STATE_NEXT_EP);

        /* Connect next_ep to the address passed in the wireup message */
        if (!(ep->state & UCP_EP_STATE_NEXT_EP_LOCAL_CONNECTED)) {
            ucs_assert(ep->state & UCP_EP_STATE_NEXT_EP);
            status = uct_ep_connect_to_ep(ucp_ep_get_stub_ep(ep)->next_ep,
                                          ucp_wireup_msg_get_addr(msg));
            if (status != UCS_OK) {
                ucs_debug("failed to connect"); /* TODO send reject */
                return;
            }

            ucs_debug("connected next ep %p", ucp_ep_get_stub_ep(ep)->next_ep);
            ep->state |= UCP_EP_STATE_NEXT_EP_LOCAL_CONNECTED;
        }

        /* Send a reply, which also includes the address of next_ep */
        status = ucp_ep_wireup_send(ep, UCP_WIREUP_FLAG_REPLY|UCP_WIREUP_FLAG_ADDR,
                                    msg->src_rsc_index, -1, 0);
        if (status != UCS_OK) {
            return;
        }

        ucs_memory_cpu_fence();
        ep->state |= UCP_EP_STATE_WIREUP_REPLY_SENT;
    }
}

static void ucp_wireup_process_reply(ucp_worker_h worker, ucp_ep_h ep,
                                     ucp_wireup_msg_t *msg)
{
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

    if (msg->dst_rsc_index != ep->rsc_index) {
        ucs_error("got connection reply on a different resource");
        return;
    }

    /*
     * If we got a reply, it means the remote side got our address and connected
     * to us.
     */
    ep->state |= UCP_EP_STATE_NEXT_EP_REMOTE_CONNECTED;

    /* If we have not connected yet, do it now */
    if (!(ep->state & UCP_EP_STATE_NEXT_EP_LOCAL_CONNECTED)) {
        ucs_assert(ep->state & UCP_EP_STATE_NEXT_EP);
        status = uct_ep_connect_to_ep(ucp_ep_get_stub_ep(ep)->next_ep,
                                      ucp_wireup_msg_addr(msg));
        if (status != UCS_OK) {
            ucs_error("failed to connect");
            return;
        }

        ucs_debug("connected next ep %p", ucp_ep_get_stub_ep(ep)->next_ep);
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
    status = ucp_ep_wireup_send(ep, UCP_WIREUP_FLAG_ACK, msg->src_rsc_index, -1, 0);
    if (status != UCS_OK) {
        return;
    }

    ucs_memory_cpu_fence();
    ep->state |= UCP_EP_STATE_WIREUP_ACK_SENT;
}

static void ucp_wireup_process_ack(ucp_worker_h worker, ucp_ep_h ep,
                                   ucp_wireup_msg_t *msg)
{
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
    ucp_ep_h ep;

    UCS_ASYNC_BLOCK(&worker->async);

    ep = ucp_worker_ep_find(worker, msg->src_uuid);
    if (msg->flags & UCP_WIREUP_FLAG_REQUSET) {
        ucp_wireup_process_request(worker, ep, msg);
    } else if (msg->flags & UCP_WIREUP_FLAG_REPLY) {
        ucp_wireup_process_reply(worker, ep, msg);
    } else if (msg->flags & UCP_WIREUP_FLAG_ACK) {
        ucp_wireup_process_ack(worker, ep, msg);
    } else {
        ucs_bug("invalid wireup message");
    }

    UCS_ASYNC_UNBLOCK(&worker->async);
    return UCS_OK;
}

ucs_status_t ucp_wireup_start(ucp_ep_h ep, ucp_address_t *address)
{
    ucp_worker_h worker = ep->worker;
    uct_iface_addr_t *uct_addr, *aux_addr;
    uct_iface_attr_t *iface_attr;
    ucp_rsc_index_t dst_rsc_index, dst_aux_rsc_index;
    ucp_rsc_index_t aux_rsc_index;
    ucp_rsc_index_t dst_aux_pd_index;
    uint64_t reachable_pds;
    ucs_status_t status;

    UCS_ASYNC_BLOCK(&worker->async);

    /*
     * Select best transport for runtime
     */
    status = ucp_select_transport(worker, address, ucp_runtime_score_func,
                                  &ep->rsc_index, &dst_rsc_index,
                                  &ep->dst_pd_index, &uct_addr,
                                  &reachable_pds,
                                  "runtime");
    if (status != UCS_OK) {
        ucs_debug("No suitable transport found");
        goto err;
    }

    iface_attr = &worker->iface_attrs[ep->rsc_index];

    /*
     * If the selected transport can be connected directly, do it.
     */
    if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
        status = uct_ep_create_connected(worker->ifaces[ep->rsc_index],
                                         uct_addr, &ep->uct_ep);
        if (status != UCS_OK) {
            ucs_debug("failed to create ep");
            goto err;
        }

        ucs_debug("created connected ep %p to %s using " UCT_TL_RESOURCE_DESC_FMT,
                  ep->uct_ep, ucp_ep_peer_name(ep),
                  UCT_TL_RESOURCE_DESC_ARG(&worker->context->tl_rscs[ep->rsc_index].tl_rsc));
        ucp_wireup_ep_ready_to_send(ep);
        goto out;
    } else if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP)) {
        status = UCS_ERR_UNREACHABLE;
        goto err;
    }

    /*
     * If we cannot connect the selected transport directly, select another
     * transport to be an auxiliary.
     */
    status = ucp_select_transport(worker, address, ucp_aux_score_func,
                                  &aux_rsc_index, &dst_aux_rsc_index,
                                  &dst_aux_pd_index, &aux_addr,
                                  &reachable_pds,
                                  "auxiliary");
    if (status != UCS_OK) {
        goto err;
    }

    status = ucp_wireup_start_aux(ep, aux_addr, aux_rsc_index);
    if (status != UCS_OK) {
        goto err;
    }

    /* Send initial connection request for wiring-up the transport */
    status = ucp_ep_wireup_send(ep, UCP_WIREUP_FLAG_REQUSET|UCP_WIREUP_FLAG_ADDR|
                                    UCP_WIREUP_FLAG_AUX_ADDR,
                                dst_rsc_index, dst_aux_rsc_index, 1);
    if (status != UCS_OK) {
        goto err_stop_aux;
    }

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return UCS_OK;

err_stop_aux:
    ucp_wireup_stop_aux(ep);
err:
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
    status = ucp_ep_wireup_send(ep, UCP_WIREUP_FLAG_REQUSET|UCP_WIREUP_FLAG_ADDR,
                                -1, -1, 1);
    return status;
}

void ucp_address_peer_name(ucp_address_t *address, char *name)
{
    uint8_t length = *(uint8_t*)(address + sizeof(uint64_t));
    ucs_assertv(length < UCP_PEER_NAME_MAX, "length=%d", length);
    memcpy(name, address + sizeof(uint64_t) + 1, length);
    name[length] = '\0';
}

UCP_DEFINE_AM(-1, UCP_AM_ID_WIREUP, ucp_wireup_msg_handler, 
              ucp_wireup_msg_dump, UCT_AM_CB_FLAG_ASYNC);


