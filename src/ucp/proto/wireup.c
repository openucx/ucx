/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ucp_int.h"

#include <ucs/type/class.h>


/**
 * Dummy endpoint, to hold off send requests until wireup process completes.
 */
typedef struct ucp_dummy_ep {
    uct_ep_t          super;
    ucp_ep_h          ep;
    uct_iface_t       iface;
    volatile uint32_t refcount;
} ucp_dummy_ep_t;

static UCS_CLASS_DEFINE_DELETE_FUNC(ucp_dummy_ep_t, uct_ep_t);

static ucs_status_t ucp_dummy_ep_send_func(uct_ep_h uct_ep)
{
    ucp_dummy_ep_t *dummy_ep = ucs_derived_of(uct_ep, ucp_dummy_ep_t);
    ucp_ep_h ep = dummy_ep->ep;

    /*
     * We switch the endpoint in this function (instead in wireup code) since
     * this is guaranteed to run from the main thread.
     */
    sched_yield();
    ucs_async_check_miss(&ep->worker->async);
    if (ep->state & UCP_EP_STATE_REMOTE_CONNECTED) {
        ep->uct.ep      = ep->uct.next_ep;
        ep->uct.next_ep = NULL;
        uct_ep_destroy(&dummy_ep->super);
    }
    return UCS_ERR_NO_RESOURCE;
}

UCS_CLASS_INIT_FUNC(ucp_dummy_ep_t, ucp_ep_h ucp_ep) {

    memset(&self->iface, 0, sizeof(self->iface));
    self->iface.ops.ep_flush          = (void*)ucs_empty_function_return_success;
    self->iface.ops.ep_destroy        = UCS_CLASS_DELETE_FUNC_NAME(ucp_dummy_ep_t);
    self->iface.ops.ep_req_notify     = (void*)ucs_empty_function_return_success;
    self->iface.ops.ep_put_short      = (void*)ucp_dummy_ep_send_func;
    self->iface.ops.ep_put_bcopy      = (void*)ucp_dummy_ep_send_func;
    self->iface.ops.ep_put_zcopy      = (void*)ucp_dummy_ep_send_func;
    self->iface.ops.ep_get_bcopy      = (void*)ucp_dummy_ep_send_func;
    self->iface.ops.ep_get_zcopy      = (void*)ucp_dummy_ep_send_func;
    self->iface.ops.ep_am_short       = (void*)ucp_dummy_ep_send_func;
    self->iface.ops.ep_am_bcopy       = (void*)ucp_dummy_ep_send_func;
    self->iface.ops.ep_am_zcopy       = (void*)ucp_dummy_ep_send_func;
    self->iface.ops.ep_atomic_add64   = (void*)ucp_dummy_ep_send_func;
    self->iface.ops.ep_atomic_fadd64  = (void*)ucp_dummy_ep_send_func;
    self->iface.ops.ep_atomic_swap64  = (void*)ucp_dummy_ep_send_func;
    self->iface.ops.ep_atomic_cswap64 = (void*)ucp_dummy_ep_send_func;
    self->iface.ops.ep_atomic_add32   = (void*)ucp_dummy_ep_send_func;
    self->iface.ops.ep_atomic_fadd32  = (void*)ucp_dummy_ep_send_func;
    self->iface.ops.ep_atomic_swap32  = (void*)ucp_dummy_ep_send_func;
    self->iface.ops.ep_atomic_cswap32 = (void*)ucp_dummy_ep_send_func;

    UCS_CLASS_CALL_SUPER_INIT(uct_ep_t, &self->iface);

    self->ep       = ucp_ep;
    self->refcount = 1;
    return UCS_OK;
}

UCS_CLASS_CLEANUP_FUNC(ucp_dummy_ep_t) {
}

UCS_CLASS_DEFINE(ucp_dummy_ep_t, uct_ep_t);

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


static uint64_t ucp_address_uuid(ucp_address_t *address)
{
    return *(uint64_t*)address;
}

static void ucp_address_iter_init(ucp_address_t *address, void **iter)
{
    *iter = (void*)( (uint64_t*)address + 1);
}

static int ucp_address_iter_next(void **iter, struct sockaddr **iface_addr_p,
                                 char *tl_name, ucp_rsc_index_t *pd_index)
{
    char *ptr = *iter;
    uint8_t iface_addr_len;
    uint8_t tl_name_len;

    iface_addr_len = *(uint8_t*)(ptr++);
    if (iface_addr_len == 0) {
        return 0;
    }

    *iface_addr_p  = (struct sockaddr*)ptr;
    ptr += iface_addr_len;

    tl_name_len = *(uint8_t*)(ptr++);
    ucs_assert_always(tl_name_len < UCT_TL_NAME_MAX);
    memcpy(tl_name, ptr, tl_name_len);
    tl_name[tl_name_len] = '\0';
    ptr += tl_name_len;

    *pd_index = *((ucp_rsc_index_t*)ptr++);

    *iter = ptr;
    return 1;
}

static void ucp_ep_remote_connected(ucp_ep_h ep)
{
    ucp_worker_h worker          = ep->worker;
    uct_iface_attr_t *iface_attr = &worker->iface_attrs[ep->uct.rsc_index];

    ucs_debug("connected 0x%"PRIx64"->0x%"PRIx64, worker->uuid, ep->dest_uuid);

    ep->config.max_short_tag = iface_attr->cap.am.max_short - sizeof(uint64_t);
    ep->config.max_short_put = iface_attr->cap.put.max_short;
    ep->config.max_bcopy_put = iface_attr->cap.put.max_bcopy;

    /* Synchronize with other threads */
    ucs_memory_cpu_store_fence();

    ucs_assert_always(ep->state & UCP_EP_STATE_LOCAL_CONNECTED);
    ep->state |= UCP_EP_STATE_REMOTE_CONNECTED;
}

static void ucp_wireup_log(ucp_worker_h worker, uint8_t am_id,
                           ucp_wireup_msg_t *msg, int is_send)
{
    ucp_context_h context = worker->context;
    const char *msg_type;

    switch (am_id) {
    case UCP_AM_ID_CONN_REQ:
        msg_type = "CONN_REQ";
        break;
    case UCP_AM_ID_CONN_REP:
        msg_type = "CONN_REP";
        break;
    case UCP_AM_ID_CONN_ACK:
        msg_type = "CONN_ACK";
        break;
    default:
        return;
    }

    if (is_send) {
        ucs_trace_data("TX: %s [uuid 0x%"PRIx64" from "UCT_TL_RESOURCE_DESC_FMT" pd %s to %d af %d]",
                        msg_type, msg->src_uuid,
                        UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[msg->src_rsc_index].tl_rsc),
                        context->pd_rscs[msg->src_pd_index].pd_name,
                        msg->dst_rsc_index,
                        ((struct sockaddr*)(msg + 1))->sa_family);
    } else {
        ucs_trace_data("RX: %s [uuid 0x%"PRIx64" from %d pd %d to "UCT_TL_RESOURCE_DESC_FMT" af %d]",
                        msg_type, msg->src_uuid, msg->src_rsc_index, msg->src_pd_index,
                        UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[msg->dst_rsc_index].tl_rsc),
                        ((struct sockaddr*)(msg + 1))->sa_family);
    }
}

static ucs_status_t ucp_ep_send_wireup_am(ucp_ep_h ep, uct_ep_h uct_ep,
                                          uint8_t am_id, ucp_rsc_index_t dst_rsc_index)
{
    ucp_rsc_index_t rsc_index    = ep->uct.rsc_index;
    ucp_worker_h worker          = ep->worker;
    ucp_context_h context        = worker->context;
    uct_iface_attr_t *iface_attr = &worker->iface_attrs[rsc_index];
    ucp_wireup_msg_t *msg;
    ucs_status_t status;
    size_t msg_len;

    msg_len = sizeof(*msg) + iface_attr->ep_addr_len;

    /* TODO use custom pack callback to avoid this allocation and memcopy */
    msg = ucs_malloc(msg_len, "conn_req");
    if (msg == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    msg->src_uuid      = worker->uuid;
    msg->src_pd_index  = context->tl_rscs[rsc_index].pd_index;
    msg->src_rsc_index = rsc_index;
    msg->dst_rsc_index = dst_rsc_index;

    status = uct_ep_get_address(ep->uct.next_ep, (struct sockaddr *)(msg + 1));
    if (status != UCS_OK) {
        goto err_free;
    }

    status = uct_ep_am_bcopy(uct_ep, am_id, (uct_pack_callback_t)memcpy,
                             msg, msg_len);
    if (status != UCS_OK) {
        ucs_debug("failed to send conn msg: %s%s", ucs_status_string(status),
                  (status == UCS_ERR_NO_RESOURCE) ? ", will retry" : "");
        goto err_free;
    }

    ucp_wireup_log(worker, am_id, msg, 1);
    ucs_free(msg);
    return UCS_OK;

err_free:
    ucs_free(msg);
err:
    return status;
}

static ucs_status_t ucp_ep_wireup_op_progress(ucp_ep_h ep, ucp_ep_pending_op_t *op,
                                              uct_ep_h *uct_ep_p)
{
    ucp_ep_wireup_op_t *wireup_op = ucs_derived_of(op, ucp_ep_wireup_op_t);
    uct_ep_h uct_ep;

    switch (wireup_op->am_id) {
    case UCP_AM_ID_CONN_REQ:
    case UCP_AM_ID_CONN_REP:
        uct_ep = ep->wireup_ep;
        break;
    case UCP_AM_ID_CONN_ACK:
        uct_ep = ep->uct.next_ep;
        break;
    default:
        return UCS_ERR_INVALID_PARAM;
    }

    *uct_ep_p = uct_ep;
    return ucp_ep_send_wireup_am(ep, uct_ep, wireup_op->am_id,
                                 wireup_op->dest_rsc_index);
}

static ucs_status_t ucp_ep_wireup_send(ucp_ep_h ep, uct_ep_h uct_ep, uint8_t am_id,
                                       ucp_rsc_index_t dst_rsc_index)
{
    ucp_ep_wireup_op_t *wireup_op;
    ucs_status_t status;

    status = ucp_ep_send_wireup_am(ep, uct_ep, am_id, dst_rsc_index);
    if (status != UCS_ERR_NO_RESOURCE) {
        return status;
    }

    wireup_op = ucs_malloc(sizeof(*wireup_op), "wireup_op");
    if (wireup_op == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    wireup_op->super.progress = ucp_ep_wireup_op_progress;
    wireup_op->am_id          = am_id;
    wireup_op->dest_rsc_index = dst_rsc_index;
    ucp_ep_add_pending_op(ep, uct_ep, &wireup_op->super);
    return UCS_OK;
}

static ucs_status_t ucp_wireup_conn_req_handler(void *arg, void *data,
                                                size_t length, void *desc)
{
    ucp_worker_h worker   = arg;
    ucp_wireup_msg_t *msg = data;
    ucs_status_t status;
    ucp_ep_h ep;

    UCS_ASYNC_BLOCK(&worker->async);

    ucp_wireup_log(worker, UCP_AM_ID_CONN_REQ, msg, 0);

    ep = ucp_worker_find_ep(worker, msg->src_uuid);
    if (ep == NULL) {
        ucs_debug("ignoring connection request - not exists");
        goto out;
    }

    if (ep->state & UCP_EP_STATE_LOCAL_CONNECTED) {
         ucs_debug("ignoring connection request - already connected");
         /* TODO allow active-passive connection establishment */
         goto out;
    }

    if (ep->uct.rsc_index != msg->dst_rsc_index) {
        ucs_error("got connection request on a different resource (got: %d, expected: %d)",
                  msg->dst_rsc_index, ep->uct.rsc_index);
        /* TODO send reject, and use different transport */
        goto out;
    }

    status = uct_ep_connect_to_ep(ep->uct.next_ep, (struct sockaddr *)(msg + 1));
    if (status != UCS_OK) {
        ucs_debug("failed to connect");
        /* TODO send reject */
        goto out;
    }

    ep->state |= UCP_EP_STATE_LOCAL_CONNECTED;

    status = ucp_ep_wireup_send(ep, ep->wireup_ep, UCP_AM_ID_CONN_REP,
                                msg->src_rsc_index);
    if (status != UCS_OK) {
        goto out;
    }

    ep->state |= UCP_EP_STATE_CONN_REP_SENT;

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return UCS_OK;
}

static ucs_status_t ucp_wireup_conn_rep_handler(void *arg, void *data,
                                                size_t length, void *desc)
{
    ucp_worker_h worker   = arg;
    ucp_wireup_msg_t *msg = data;
    ucs_status_t status;
    ucp_ep_h ep;

    ucp_wireup_log(worker, UCP_AM_ID_CONN_REP, msg, 0);

    UCS_ASYNC_BLOCK(&worker->async);

    ep = ucp_worker_find_ep(worker, msg->src_uuid);
    if (ep == NULL) {
        ucs_debug("ignoring connection request - not exists");
        goto out;
    }

    if (msg->dst_rsc_index != ep->uct.rsc_index) {
        ucs_error("got connection reply on a different resource");
        goto out;
    }

    if (ep->state & UCP_EP_STATE_REMOTE_CONNECTED) {
        ucs_debug("ignoring conn_rep - remote already connected");
        goto out;
    }

    /* If we have not connected yet, do it now */
    if (!(ep->state & UCP_EP_STATE_LOCAL_CONNECTED)) {
        status = uct_ep_connect_to_ep(ep->uct.next_ep, (struct sockaddr *)(msg + 1));
        if (status != UCS_OK) {
            ucs_debug("failed to connect");
            goto out;
        }

        ep->state |= UCP_EP_STATE_LOCAL_CONNECTED;
    }

    /*
     * Send ACK to let remote side know it can start sending.
     * If we already sent a reply to the remote side, no need to send an ACK.
     *
     * We can use the new ep even from async thread, because main thread has not
     * started using it yet.
     */
    if (!(ep->state & (UCP_EP_STATE_CONN_REP_SENT|UCP_EP_STATE_CONN_ACK_SENT))) {
        status = ucp_ep_wireup_send(ep, ep->uct.next_ep, UCP_AM_ID_CONN_ACK,
                                    msg->src_rsc_index);
        if (status != UCS_OK) {
            goto out;
        }

        ep->state |= UCP_EP_STATE_CONN_ACK_SENT;
    }

    /*
     * If we got CONN_REP, it means the remote side got our address and connected
     * to us.
     */
    ucp_ep_remote_connected(ep);

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return UCS_OK;
}

static ucs_status_t ucp_wireup_conn_ack_handler(void *arg, void *data,
                                                size_t length, void *desc)
{
    ucp_worker_h worker   = arg;
    ucp_wireup_msg_t *msg = data;
    ucp_ep_h ep;

    UCS_ASYNC_BLOCK(&worker->async);

    ucp_wireup_log(worker, UCP_AM_ID_CONN_ACK, msg, 0);

    ep = ucp_worker_find_ep(worker, msg->src_uuid);
    if (ep == NULL) {
        ucs_debug("ignoring connection request - not exists");
        goto out;
    }

    if (ep->state & UCP_EP_STATE_REMOTE_CONNECTED) {
        ucs_debug("ignoring conn_ack - remote already connected");
        goto out;
    }

    /*
     * If we got CONN_ACK, it means remote side got our reply, and also
     * connected to us.
     */
    ucp_ep_remote_connected(ep);

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return UCS_OK;
}

static double ucp_wireup_score_func(uct_tl_resource_desc_t *resource,
                                    uct_iface_h iface,
                                    uct_iface_attr_t *iface_attr)
{
    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_BCOPY) ||
        !(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE))
    {
        return 0.0;
    }

    return (1e6 / resource->latency) +
           (1e3 * ucs_max(iface_attr->cap.am.max_bcopy, iface_attr->cap.am.max_short));
}

static double ucp_am_short_score_func(uct_tl_resource_desc_t *resource,
                                      uct_iface_h iface, uct_iface_attr_t *iface_attr)
{
    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_SHORT) ||
        !(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_THREAD_SINGLE))
    {
        return 0.0;
    }

    return (1e6 / resource->latency);
}

static ucs_status_t ucp_pick_best_wireup(ucp_worker_h worker, ucp_address_t *address,
                                         ucp_wireup_score_function_t score_func,
                                         ucp_rsc_index_t *src_rsc_index_p,
                                         ucp_rsc_index_t *dst_rsc_index_p,
                                         ucp_rsc_index_t *dst_pd_index_p,
                                         struct sockaddr **addr_p,
                                         uint64_t *reachable_pds,
                                         const char *title)
{
    ucp_context_h context = worker->context;
    ucp_rsc_index_t src_rsc_index, dst_rsc_index;
    ucp_rsc_index_t pd_index;
    struct sockaddr *addr, *best_addr;
    double score, best_score;
    uct_iface_attr_t *iface_attr;
    uct_tl_resource_desc_t *resource;
    char tl_name[UCT_TL_NAME_MAX];
    uct_iface_h iface;
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
    dst_rsc_index = 0;
    ucp_address_iter_init(address, &iter);
    while (ucp_address_iter_next(&iter, &addr, tl_name, &pd_index)) {

        for (src_rsc_index = 0; src_rsc_index < context->num_tls; ++src_rsc_index) {
            resource   = &context->tl_rscs[src_rsc_index].tl_rsc;
            iface      = worker->ifaces[src_rsc_index];
            iface_attr = &worker->iface_attrs[src_rsc_index];

            /* Must be reachable address, on same transport */
            if (strcmp(tl_name, resource->tl_name) ||
                !uct_iface_is_reachable(iface, addr))
            {
                continue;
            }

            *reachable_pds |= UCS_BIT(pd_index);

            score = score_func(resource, iface, iface_attr);
            ucs_trace("%s " UCT_TL_RESOURCE_DESC_FMT " score %.2f",
                      title, UCT_TL_RESOURCE_DESC_ARG(resource), score);
            if (score > best_score) {
                ucs_assert(addr != NULL);
                best_score       = score;
                best_addr        = addr;
                *src_rsc_index_p = src_rsc_index;
                *dst_rsc_index_p = dst_rsc_index;
                *dst_pd_index_p  = pd_index;
            }
        }

        ++dst_rsc_index;
    }

    if (best_addr == NULL) {
        return UCS_ERR_UNREACHABLE;
    }

    ucs_debug("%s: " UCT_TL_RESOURCE_DESC_FMT " to %d pd %d", title,
              UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[*src_rsc_index_p].tl_rsc),
              *dst_rsc_index_p, *dst_pd_index_p);
    *addr_p = best_addr;
    return UCS_OK;
}

ucs_status_t ucp_wireup_set_am_handlers(ucp_worker_h worker, uct_iface_h iface)
{
    ucs_status_t status;

    status = uct_iface_set_am_handler(iface, UCP_AM_ID_CONN_REQ,
                                      ucp_wireup_conn_req_handler, worker);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_iface_set_am_handler(iface, UCP_AM_ID_CONN_REP,
                                      ucp_wireup_conn_rep_handler, worker);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_iface_set_am_handler(iface, UCP_AM_ID_CONN_ACK,
                                      ucp_wireup_conn_ack_handler, worker);
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}

ucs_status_t ucp_ep_wireup_start(ucp_ep_h ep, ucp_address_t *address)
{
    ucp_worker_h worker = ep->worker;
    struct sockaddr *am_short_addr;
    ucp_rsc_index_t wireup_rsc_index;
    struct sockaddr *wireup_addr;
    uct_iface_attr_t *iface_attr;
    uct_iface_h iface;
    ucp_rsc_index_t dst_rsc_index, wireup_dst_rsc_index;
    ucp_rsc_index_t wireup_dst_pd_index;
    ucs_status_t status;

    UCS_ASYNC_BLOCK(&worker->async);

    ep->dest_uuid = ucp_address_uuid(address);
    sglib_hashed_ucp_ep_t_add(worker->ep_hash, ep);

    ucs_debug("connecting 0x%"PRIx64"->0x%"PRIx64, worker->uuid, ep->dest_uuid);

    /*
     * Select best transport for active messages
     */
    status = ucp_pick_best_wireup(worker, address, ucp_am_short_score_func,
                                  &ep->uct.rsc_index, &dst_rsc_index,
                                  &ep->uct.dst_pd_index, &am_short_addr,
                                  &ep->uct.reachable_pds,
                                  "short_am");
    if (status != UCS_OK) {
        ucs_error("No transport for short active message");
        goto err;
    }

    iface      = worker->ifaces[ep->uct.rsc_index];
    iface_attr = &worker->iface_attrs[ep->uct.rsc_index];

    /*
     * If the selected transport can be connected directly, do it.
     */
    if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
        status = uct_ep_create_connected(iface, am_short_addr, &ep->uct.next_ep);
        if (status != UCS_OK) {
            ucs_debug("failed to create ep");
            goto err;
        }

        ep->state |= UCP_EP_STATE_LOCAL_CONNECTED;
        ucp_ep_remote_connected(ep);
        goto out;
    }

    /*
     * If we cannot connect the selected transport directly, select another
     * transport for doing the wireup.
     */
    status = ucp_pick_best_wireup(worker, address, ucp_wireup_score_func,
                                  &wireup_rsc_index, &wireup_dst_rsc_index,
                                  &wireup_dst_pd_index, &wireup_addr,
                                  &ep->uct.reachable_pds,
                                  "wireup");
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_ep_create_connected(worker->ifaces[wireup_rsc_index],
                                     wireup_addr, &ep->wireup_ep);
    if (status != UCS_OK) {
        goto err;
    }

    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP)) {
        status = UCS_ERR_UNREACHABLE;
        goto err_destroy_wireup_ep;
    }

    /*
     * Until the transport is connected, send operations should return NO_RESOURCE.
     * Plant a dummy endpoint object which will do it.
     */
    status = UCS_CLASS_NEW(ucp_dummy_ep_t, &ep->uct.ep, ep);
    if (status != UCS_OK) {
        goto err_destroy_wireup_ep;
    }

    /*
     * Create endpoint for the transport we need to wire-up.
     */
    status = uct_ep_create(iface, &ep->uct.next_ep);
    if (status != UCS_OK) {
        goto err_destroy_uct_ep;
    }

    /*
     * Send initial connection request for wiring-up the transport.
     */
    status = ucp_ep_wireup_send(ep, ep->wireup_ep, UCP_AM_ID_CONN_REQ,
                                dst_rsc_index);
    if (status != UCS_OK) {
        goto err_destroy_next_ep;
    }

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return UCS_OK;

err_destroy_next_ep:
    uct_ep_destroy(ep->uct.next_ep);
err_destroy_uct_ep:
    uct_ep_destroy(ep->uct.ep);
err_destroy_wireup_ep:
    uct_ep_destroy(ep->wireup_ep);
err:
    sglib_hashed_ucp_ep_t_delete(worker->ep_hash, ep);
    UCS_ASYNC_UNBLOCK(&worker->async);
    return status;
}

void ucp_ep_wireup_stop(ucp_ep_h ep)
{
    ucp_worker_h worker = ep->worker;

    if (ep->uct.next_ep != NULL) {
        while (uct_ep_flush(ep->uct.next_ep) != UCS_OK) {
            ucp_worker_progress(ep->worker);
        }
        uct_ep_destroy(ep->uct.next_ep);
    }

    if (ep->wireup_ep != NULL) {
        while (uct_ep_flush(ep->wireup_ep) != UCS_OK) {
            ucp_worker_progress(ep->worker);
        }
        uct_ep_destroy(ep->wireup_ep);
    }

    UCS_ASYNC_BLOCK(&worker->async);
    sglib_hashed_ucp_ep_t_delete(worker->ep_hash, ep);
    UCS_ASYNC_UNBLOCK(&worker->async);
}
