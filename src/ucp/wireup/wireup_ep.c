/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "wireup_ep.h"
#include "wireup.h"

#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_worker.h>
#include <ucs/arch/atomic.h>
#include <ucs/datastruct/queue.h>
#include <ucs/type/class.h>
#include <ucs/sys/string.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_request.inl>


UCS_CLASS_DECLARE(ucp_wireup_ep_t, ucp_ep_h);


static UCS_CLASS_DEFINE_DELETE_FUNC(ucp_wireup_ep_t, uct_ep_t);

static inline ucs_queue_elem_t* ucp_wireup_ep_req_priv(uct_pending_req_t *req)
{
    UCS_STATIC_ASSERT(sizeof(ucs_queue_elem_t) <= UCT_PENDING_REQ_PRIV_LEN);
    return (ucs_queue_elem_t*)req->priv;
}

static ucs_status_t
ucp_wireup_ep_connect_to_ep(uct_ep_h uct_ep, const uct_device_addr_t *dev_addr,
                            const uct_ep_addr_t *ep_addr)
{
    ucp_wireup_ep_t *wireup_ep = ucs_derived_of(uct_ep, ucp_wireup_ep_t);
    wireup_ep->flags |= UCP_WIREUP_EP_FLAG_LOCAL_CONNECTED;
    return uct_ep_connect_to_ep(wireup_ep->super.uct_ep, dev_addr, ep_addr);
}

/*
 * We switch the endpoint in this function (instead in wireup code) since
 * this is guaranteed to run from the main thread.
 */
static unsigned ucp_wireup_ep_progress(void *arg)
{
    ucp_wireup_ep_t *wireup_ep = arg;
    ucp_ep_h ucp_ep = wireup_ep->super.ucp_ep;
    ucs_queue_head_t tmp_pending_queue;
    uct_pending_req_t *uct_req;
    ucp_request_t *req;

    UCS_ASYNC_BLOCK(&ucp_ep->worker->async);

    ucs_assert(wireup_ep->flags & UCP_WIREUP_EP_FLAG_READY);
    ucs_assert(wireup_ep->super.uct_ep != NULL);

    /* If we still have pending wireup messages, send them out first */
    if (wireup_ep->pending_count != 0) {
        goto out;
    }

    ucs_trace("ep %p: switching wireup_ep %p to ready state", ucp_ep, wireup_ep);

    /* Move wireup pending queue to temporary queue and remove references to
     * the wireup progress function
     */
    ucs_queue_head_init(&tmp_pending_queue);
    ucs_queue_for_each_extract(uct_req, &wireup_ep->pending_q, priv, 1) {
        ucs_queue_push(&tmp_pending_queue, ucp_wireup_ep_req_priv(uct_req));
    }

    /* Switch to real transport and destroy proxy endpoint (aux_ep as well) */
    ucp_proxy_ep_replace(&wireup_ep->super);
    wireup_ep = NULL;

    /* Replay pending requests */
    ucs_queue_for_each_extract(uct_req, &tmp_pending_queue, priv, 1) {
        req = ucs_container_of(uct_req, ucp_request_t, send.uct);
        ucs_assert(req->send.ep == ucp_ep);
        ucp_request_send(req);
        --ucp_ep->worker->wireup_pend_count;
    }

out:
    UCS_ASYNC_UNBLOCK(&ucp_ep->worker->async);
    return 0;
}

static ssize_t ucp_wireup_ep_bcopy_send_func(uct_ep_h uct_ep)
{
    return UCS_ERR_NO_RESOURCE;
}

static uct_ep_h ucp_wireup_ep_get_msg_ep(ucp_wireup_ep_t *wireup_ep)
{
    uct_ep_h wireup_msg_ep;

    if (wireup_ep->flags & UCP_WIREUP_EP_FLAG_READY) {
        wireup_msg_ep = wireup_ep->super.uct_ep;
    } else {
        wireup_msg_ep = wireup_ep->aux_ep;
    }
    ucs_assertv(wireup_msg_ep != NULL,
                "ucp_ep=%p wireup_ep=%p flags=%c%c next_ep=%p aux_ep=%p",
                wireup_ep->super.ucp_ep, wireup_ep,
                (wireup_ep->flags & UCP_WIREUP_EP_FLAG_LOCAL_CONNECTED) ? 'c' : '-',
                (wireup_ep->flags & UCP_WIREUP_EP_FLAG_READY)           ? 'r' : '-',
                wireup_ep->super.uct_ep, wireup_ep->aux_ep);
    return wireup_msg_ep;
}

static ucs_status_t ucp_wireup_ep_progress_pending(uct_pending_req_t *self)
{
    ucp_request_t *proxy_req = ucs_container_of(self, ucp_request_t, send.uct);
    uct_pending_req_t *req = proxy_req->send.proxy.req;
    ucp_wireup_ep_t *wireup_ep = proxy_req->send.proxy.wireup_ep;
    ucs_status_t status;

    status = req->func(req);
    if (status == UCS_OK) {
        ucs_atomic_add32(&wireup_ep->pending_count, -1);
        ucp_request_put(proxy_req);
    }
    return status;
}

static void
ucp_wireup_ep_pending_req_release(uct_pending_req_t *self, void *arg)
{
    ucp_request_t   *proxy_req = ucs_container_of(self, ucp_request_t,
                                                  send.uct);
    ucp_wireup_ep_t *wireup_ep = proxy_req->send.proxy.wireup_ep;
    ucp_request_t   *req;

    ucs_atomic_add32(&wireup_ep->pending_count, -1);
 
    if (proxy_req->send.proxy.req->func == ucp_wireup_msg_progress) {
        req = ucs_container_of(proxy_req->send.proxy.req, ucp_request_t,
                               send.uct);
        ucs_free((void*)req->send.buffer);
        ucs_free(req);
    }

    ucp_request_put(proxy_req);
}

static ucs_status_t ucp_wireup_ep_pending_add(uct_ep_h uct_ep,
                                              uct_pending_req_t *req)
{
    ucp_wireup_ep_t *wireup_ep = ucs_derived_of(uct_ep, ucp_wireup_ep_t);
    ucp_ep_h ucp_ep = wireup_ep->super.ucp_ep;
    ucp_worker_h worker = ucp_ep->worker;
    ucp_request_t *proxy_req;
    uct_ep_h wireup_msg_ep;
    ucs_status_t status;

    UCS_ASYNC_BLOCK(&worker->async);
    if (req->func == ucp_wireup_msg_progress) {
        proxy_req = ucs_mpool_get(&worker->req_mp);
        if (proxy_req == NULL) {
            status = UCS_ERR_NO_MEMORY;
            goto out;
        }

        wireup_msg_ep = ucp_wireup_ep_get_msg_ep(wireup_ep);

        proxy_req->send.uct.func            = ucp_wireup_ep_progress_pending;
        proxy_req->send.proxy.req           = req;
        proxy_req->send.proxy.wireup_ep     = wireup_ep;
        proxy_req->send.state.uct_comp.func = NULL;

        status = uct_ep_pending_add(wireup_msg_ep, &proxy_req->send.uct);
        if (status == UCS_OK) {
            ucs_atomic_add32(&wireup_ep->pending_count, +1);
        } else {
            ucp_request_put(proxy_req);
        }
    } else {
        ucs_queue_push(&wireup_ep->pending_q, ucp_wireup_ep_req_priv(req));
        ++ucp_ep->worker->wireup_pend_count;
        status = UCS_OK;
    }
out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return status;
}

static void
ucp_wireup_ep_pending_purge(uct_ep_h uct_ep, uct_pending_purge_callback_t cb,
                            void *arg)
{
    ucp_wireup_ep_t *wireup_ep = ucs_derived_of(uct_ep, ucp_wireup_ep_t);
    uct_pending_req_t *req;
    ucp_request_t *ucp_req;

    ucs_queue_for_each_extract(req, &wireup_ep->pending_q, priv, 1) {
        ucp_req = ucs_container_of(req, ucp_request_t, send.uct);
        cb(&ucp_req->send.uct, arg);
    }

    if (wireup_ep->aux_ep != NULL) {
        uct_ep_pending_purge(wireup_ep->aux_ep, ucp_wireup_ep_pending_req_release,
                             arg);
    }
}

static ssize_t ucp_wireup_ep_am_bcopy(uct_ep_h uct_ep, uint8_t id,
                                      uct_pack_callback_t pack_cb, void *arg,
                                      unsigned flags)
{
    ucp_wireup_ep_t *wireup_ep = ucs_derived_of(uct_ep, ucp_wireup_ep_t);

    if (id == UCP_AM_ID_WIREUP) {
        return uct_ep_am_bcopy(ucp_wireup_ep_get_msg_ep(wireup_ep),
                               UCP_AM_ID_WIREUP, pack_cb, arg, flags);
    }

    return UCS_ERR_NO_RESOURCE;
}


UCS_CLASS_DEFINE_NAMED_NEW_FUNC(ucp_wireup_ep_create, ucp_wireup_ep_t, uct_ep_t,
                                ucp_ep_h);

static ucs_status_t
ucp_wireup_ep_connect_aux(ucp_wireup_ep_t *wireup_ep,
                          const ucp_ep_params_t *params, unsigned address_count,
                          const ucp_address_entry_t *address_list)
{
    ucp_ep_h ucp_ep     = wireup_ep->super.ucp_ep;
    ucp_worker_h worker = ucp_ep->worker;
    const ucp_address_entry_t *aux_addr;
    ucp_rsc_index_t rsc_index;
    unsigned aux_addr_index;
    ucs_status_t status;

    /* select an auxiliary transport which would be used to pass connection
     * establishment messages.
     */
    status = ucp_wireup_select_aux_transport(ucp_ep, params, address_list,
                                             address_count, &rsc_index,
                                             &aux_addr_index);
    if (status != UCS_OK) {
        return status;
    }

    wireup_ep->aux_rsc_index = rsc_index;
    aux_addr = &address_list[aux_addr_index];

    /* create auxiliary endpoint connected to the remote iface. */
    status = uct_ep_create_connected(worker->ifaces[rsc_index].iface,
                                     aux_addr->dev_addr, aux_addr->iface_addr,
                                     &wireup_ep->aux_ep);
    if (status != UCS_OK) {
        return status;
    }

    ucp_worker_iface_progress_ep(&worker->ifaces[rsc_index]);

    ucs_debug("ep %p: wireup_ep %p created aux_ep %p to %s using "
              UCT_TL_RESOURCE_DESC_FMT, ucp_ep, wireup_ep, wireup_ep->aux_ep,
              ucp_ep_peer_name(ucp_ep),
              UCT_TL_RESOURCE_DESC_ARG(&worker->context->tl_rscs[rsc_index].tl_rsc));
    return UCS_OK;
}

static ucs_status_t ucp_wireup_ep_flush(uct_ep_h uct_ep, unsigned flags,
                                        uct_completion_t *comp)
{
    ucp_wireup_ep_t *wireup_ep = ucs_derived_of(uct_ep, ucp_wireup_ep_t);

    if (flags & UCT_FLUSH_FLAG_CANCEL) {
        if (wireup_ep->aux_ep) {
            return uct_ep_flush(wireup_ep->aux_ep, flags, comp);
        }
        return UCS_OK;
    }
    return UCS_ERR_NO_RESOURCE;
}


UCS_CLASS_INIT_FUNC(ucp_wireup_ep_t, ucp_ep_h ucp_ep)
{
    static uct_iface_ops_t ops = {
        .ep_connect_to_ep     = ucp_wireup_ep_connect_to_ep,
        .ep_flush             = ucp_wireup_ep_flush,
        .ep_destroy           = UCS_CLASS_DELETE_FUNC_NAME(ucp_wireup_ep_t),
        .ep_pending_add       = ucp_wireup_ep_pending_add,
        .ep_pending_purge     = ucp_wireup_ep_pending_purge,
        .ep_put_short         = (void*)ucs_empty_function_return_no_resource,
        .ep_put_bcopy         = (void*)ucp_wireup_ep_bcopy_send_func,
        .ep_put_zcopy         = (void*)ucs_empty_function_return_no_resource,
        .ep_get_bcopy         = (void*)ucs_empty_function_return_no_resource,
        .ep_get_zcopy         = (void*)ucs_empty_function_return_no_resource,
        .ep_am_short          = (void*)ucs_empty_function_return_no_resource,
        .ep_am_bcopy          = ucp_wireup_ep_am_bcopy,
        .ep_am_zcopy          = (void*)ucs_empty_function_return_no_resource,
        .ep_tag_eager_short   = (void*)ucs_empty_function_return_no_resource,
        .ep_tag_eager_bcopy   = (void*)ucp_wireup_ep_bcopy_send_func,
        .ep_tag_eager_zcopy   = (void*)ucs_empty_function_return_no_resource,
        .ep_tag_rndv_zcopy    = (void*)ucs_empty_function_return_ptr_no_resource,
        .ep_tag_rndv_request  = (void*)ucs_empty_function_return_no_resource,
        .ep_atomic_add64      = (void*)ucs_empty_function_return_no_resource,
        .ep_atomic_fadd64     = (void*)ucs_empty_function_return_no_resource,
        .ep_atomic_swap64     = (void*)ucs_empty_function_return_no_resource,
        .ep_atomic_cswap64    = (void*)ucs_empty_function_return_no_resource,
        .ep_atomic_add32      = (void*)ucs_empty_function_return_no_resource,
        .ep_atomic_fadd32     = (void*)ucs_empty_function_return_no_resource,
        .ep_atomic_swap32     = (void*)ucs_empty_function_return_no_resource,
        .ep_atomic_cswap32    = (void*)ucs_empty_function_return_no_resource
    };

    UCS_CLASS_CALL_SUPER_INIT(ucp_proxy_ep_t, &ops, ucp_ep, NULL, 0);

    self->aux_ep        = NULL;
    self->sockaddr_ep   = NULL;
    self->aux_rsc_index = UCP_NULL_RESOURCE;
    self->pending_count = 0;
    self->flags         = 0;
    self->progress_id   = UCS_CALLBACKQ_ID_NULL;
    ucs_queue_head_init(&self->pending_q);

    ucs_trace("ep %p: created wireup ep %p to %s ", ucp_ep, self,
              ucp_ep_peer_name(ucp_ep));
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(ucp_wireup_ep_t)
{
    ucp_ep_h ucp_ep     = self->super.ucp_ep;
    ucp_worker_h worker = ucp_ep->worker;

    ucs_assert(ucs_queue_is_empty(&self->pending_q));
    ucs_assert(self->pending_count == 0);

    ucs_debug("ep %p: destroy wireup ep %p", ucp_ep, self);

    uct_worker_progress_unregister_safe(worker->uct, &self->progress_id);
    if (self->aux_ep != NULL) {
        ucp_worker_iface_unprogress_ep(&worker->ifaces[self->aux_rsc_index]);
        uct_ep_destroy(self->aux_ep);
    }
    if (self->sockaddr_ep != NULL) {
        uct_ep_destroy(self->sockaddr_ep);
    }
}

UCS_CLASS_DEFINE(ucp_wireup_ep_t, ucp_proxy_ep_t);

ucp_rsc_index_t ucp_wireup_ep_get_aux_rsc_index(uct_ep_h uct_ep)
{
    ucp_wireup_ep_t *wireup_ep = ucs_derived_of(uct_ep, ucp_wireup_ep_t);

    if (!ucp_wireup_ep_test(uct_ep)) {
        return UCP_NULL_RESOURCE;
    }

    ucs_assert(wireup_ep->aux_ep != NULL);
    return wireup_ep->aux_rsc_index;
}

ucs_status_t ucp_wireup_ep_connect(uct_ep_h uct_ep, const ucp_ep_params_t *params,
                                   ucp_rsc_index_t rsc_index, int connect_aux,
                                   unsigned address_count,
                                   const ucp_address_entry_t *address_list)
{
    ucp_wireup_ep_t *wireup_ep = ucs_derived_of(uct_ep, ucp_wireup_ep_t);
    ucp_ep_h ucp_ep            = wireup_ep->super.ucp_ep;
    ucp_worker_h worker        = ucp_ep->worker;
    ucs_status_t status;
    uct_ep_h next_ep;

    ucs_assert(ucp_wireup_ep_test(uct_ep));

    status = uct_ep_create(worker->ifaces[rsc_index].iface, &next_ep);
    if (status != UCS_OK) {
        goto err;
    }

    ucp_proxy_ep_set_uct_ep(&wireup_ep->super, next_ep, 1);

    ucs_debug("ep %p: created next_ep %p to %s using " UCT_TL_RESOURCE_DESC_FMT,
              ucp_ep, wireup_ep->super.uct_ep, ucp_ep_peer_name(ucp_ep),
              UCT_TL_RESOURCE_DESC_ARG(&worker->context->tl_rscs[rsc_index].tl_rsc));

    /* we need to create an auxiliary transport only for active messages */
    if (connect_aux) {
        status = ucp_wireup_ep_connect_aux(wireup_ep, params, address_count,
                                           address_list);
        if (status != UCS_OK) {
            goto err_destroy_next_ep;
        }
    }

    return UCS_OK;

err_destroy_next_ep:
    uct_ep_destroy(wireup_ep->super.uct_ep);
    wireup_ep->super.uct_ep = NULL;
err:
    return status;
}

ucs_status_t ucp_wireup_ep_connect_to_sockaddr(uct_ep_h uct_ep,
                                               const ucp_ep_params_t *params)
{
    ucp_wireup_ep_t *wireup_ep = ucs_derived_of(uct_ep, ucp_wireup_ep_t);
    ucp_ep_h ucp_ep            = wireup_ep->super.ucp_ep;
    ucp_worker_h worker        = ucp_ep->worker;
    ucp_context_h context      = worker->context;
    char saddr_str[UCS_SOCKADDR_STRING_LEN];
    ucp_wireup_sockaddr_priv_t *conn_priv;
    size_t address_length, conn_priv_len;
    ucp_address_t *worker_address;
    ucp_rsc_index_t sockaddr_rsc;
    ucp_worker_iface_t *wiface;
    ucs_status_t status;

    ucs_assert(ucp_wireup_ep_test(uct_ep));

    status = ucp_wireup_select_sockaddr_transport(ucp_ep, params, &sockaddr_rsc);
    if (status != UCS_OK) {
        goto out;
     }

    status = ucp_worker_get_address(worker, &worker_address, &address_length);
    if (status != UCS_OK) {
        goto out;
    }

    conn_priv_len = sizeof(*conn_priv) + address_length;

    /* check private data limitation */
    wiface = &worker->ifaces[sockaddr_rsc];
    if (conn_priv_len > wiface->attr.max_conn_priv) {
        ucs_error("sockaddr connection priv data (%zu) exceeds max_priv on "
                  UCT_TL_RESOURCE_DESC_FMT" (%zu)", conn_priv_len,
                  UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[sockaddr_rsc].tl_rsc),
                  wiface->attr.max_conn_priv);
        status = UCS_ERR_UNREACHABLE;
        goto out_free_address;
    }

    conn_priv = ucs_malloc(conn_priv_len ,"ucp_sockaddr_conn_priv");
    if (conn_priv == NULL) {
        ucs_error("failed to allocate buffer for sockaddr conn priv");
        status = UCS_ERR_NO_MEMORY;
        goto out_free_address;
    }

    /* pack private data */
    conn_priv->err_mode = UCP_PARAM_VALUE(EP, params, err_mode, ERR_HANDLING_MODE,
                                          UCP_ERR_HANDLING_MODE_NONE);
    conn_priv->ep_uuid  = ucp_ep->dest_uuid;
    memcpy(conn_priv + 1, worker_address, address_length);

    /* send connection request using the transport */
    status = uct_ep_create_sockaddr(wiface->iface, &params->sockaddr, conn_priv,
                                    conn_priv_len, &wireup_ep->sockaddr_ep);
    if (status != UCS_OK) {
        goto out_free_priv;
    }

    ucs_debug("ep %p connecting to %s", ucp_ep,
              ucs_sockaddr_str(params->sockaddr.addr, saddr_str, sizeof(saddr_str)));
    status = UCS_OK;

out_free_priv:
    ucs_free(conn_priv);
out_free_address:
    ucp_worker_release_address(worker, worker_address);
out:
    return status;
}

void ucp_wireup_ep_set_next_ep(uct_ep_h uct_ep, uct_ep_h next_ep)
{
    ucp_wireup_ep_t *wireup_ep = ucs_derived_of(uct_ep, ucp_wireup_ep_t);

    ucs_assert(ucp_wireup_ep_test(uct_ep));
    ucs_assert(wireup_ep->super.uct_ep == NULL);
    wireup_ep->flags |= UCP_WIREUP_EP_FLAG_LOCAL_CONNECTED;
    ucp_proxy_ep_set_uct_ep(&wireup_ep->super, next_ep, 1);
}

uct_ep_h ucp_wireup_ep_extract_next_ep(uct_ep_h uct_ep)
{
    ucp_wireup_ep_t *wireup_ep = ucs_derived_of(uct_ep, ucp_wireup_ep_t);
    uct_ep_h next_ep;

    ucs_assert_always(ucp_wireup_ep_test(uct_ep));
    next_ep = wireup_ep->super.uct_ep;
    wireup_ep->super.uct_ep = NULL;
    return next_ep;
}

void ucp_wireup_ep_remote_connected(uct_ep_h uct_ep)
{
    ucp_wireup_ep_t *wireup_ep = ucs_derived_of(uct_ep, ucp_wireup_ep_t);
    ucp_ep_h ucp_ep = wireup_ep->super.ucp_ep;

    ucs_assert(ucp_wireup_ep_test(uct_ep));
    ucs_assert(wireup_ep->super.uct_ep != NULL);
    ucs_assert(wireup_ep->flags & UCP_WIREUP_EP_FLAG_LOCAL_CONNECTED);

    ucs_trace("ep %p: wireup ep %p is remote-connected", ucp_ep, wireup_ep);
    wireup_ep->flags |= UCP_WIREUP_EP_FLAG_READY;
    uct_worker_progress_register_safe(ucp_ep->worker->uct,
                                      ucp_wireup_ep_progress, wireup_ep, 0,
                                      &wireup_ep->progress_id);
    ucp_worker_signal_internal(ucp_ep->worker);
}

int ucp_wireup_ep_test(uct_ep_h uct_ep)
{
    return uct_ep->iface->ops.ep_destroy ==
                    UCS_CLASS_DELETE_FUNC_NAME(ucp_wireup_ep_t);
}

int ucp_wireup_ep_is_owner(uct_ep_h uct_ep, uct_ep_h owned_ep)
{
    ucp_wireup_ep_t *wireup_ep;

    if (!ucp_wireup_ep_test(uct_ep)) {
        return 0;
    }

    wireup_ep = ucs_derived_of(uct_ep, ucp_wireup_ep_t);
    return (wireup_ep->aux_ep == owned_ep) || (wireup_ep->sockaddr_ep == owned_ep);

}

void ucp_wireup_ep_disown(uct_ep_h uct_ep, uct_ep_h owned_ep)
{
    ucp_wireup_ep_t *wireup_ep = ucs_derived_of(uct_ep, ucp_wireup_ep_t);

    ucs_assert_always(ucp_wireup_ep_test(uct_ep));
    if (wireup_ep->aux_ep == owned_ep) {
        wireup_ep->aux_ep = NULL;
    } else if (wireup_ep->sockaddr_ep == owned_ep) {
        wireup_ep->sockaddr_ep = NULL;
    }
}
