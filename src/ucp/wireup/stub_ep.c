/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "stub_ep.h"
#include "wireup.h"

#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_worker.h>
#include <ucs/arch/atomic.h>
#include <ucs/datastruct/queue.h>
#include <ucs/type/class.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_request.inl>


typedef struct {
    uct_pending_purge_callback_t cb;
    void                         *arg;
} ucp_stub_ep_pending_release_proxy_arg_t;


UCS_CLASS_DECLARE(ucp_stub_ep_t, ucp_ep_h);


static UCS_CLASS_DEFINE_DELETE_FUNC(ucp_stub_ep_t, uct_ep_t);

static inline ucs_queue_elem_t* ucp_stub_ep_req_priv(uct_pending_req_t *req)
{
    UCS_STATIC_ASSERT(sizeof(ucs_queue_elem_t) <= UCT_PENDING_REQ_PRIV_LEN);
    return (ucs_queue_elem_t*)req->priv;
}

static ucs_status_t ucp_stub_ep_get_address(uct_ep_h uct_ep, uct_ep_addr_t *addr)
{
    ucp_stub_ep_t *stub_ep = ucs_derived_of(uct_ep, ucp_stub_ep_t);
    return uct_ep_get_address(stub_ep->next_ep, addr);
}

static ucs_status_t ucp_stub_ep_connect_to_ep(uct_ep_h uct_ep,
                                              const uct_device_addr_t *dev_addr,
                                              const uct_ep_addr_t *ep_addr)
{
    ucp_stub_ep_t *stub_ep = ucs_derived_of(uct_ep, ucp_stub_ep_t);
    stub_ep->flags |= UCP_STUB_EP_FLAG_LOCAL_CONNECTED;
    return uct_ep_connect_to_ep(stub_ep->next_ep, dev_addr, ep_addr);
}

/*
 * We switch the endpoint in this function (instead in wireup code) since
 * this is guaranteed to run from the main thread.
 */
static void ucp_stub_ep_progress(void *arg)
{
    ucp_stub_ep_t *stub_ep = arg;
    ucp_ep_h ep = stub_ep->ep;
    ucs_queue_head_t tmp_pending_queue;
    uct_pending_req_t *uct_req;
    ucp_lane_index_t lane;
    ucp_request_t *req;
    uct_ep_h uct_ep;

    UCS_ASYNC_BLOCK(&ep->worker->async);

    ucs_assert(stub_ep->flags & UCP_STUB_EP_FLAG_READY);
    ucs_assert(stub_ep->next_ep != NULL);

    /* If we still have pending wireup messages, send them out first */
    if (stub_ep->pending_count != 0) {
        UCS_ASYNC_UNBLOCK(&ep->worker->async);
        return;
    }

    ucs_trace("ep %p: switching stub_ep %p to ready state", ep, stub_ep);

    /* Take out next_ep */
    uct_ep = stub_ep->next_ep;
    stub_ep->next_ep = NULL;

    /* Move stub pending queue to temporary queue and remove references to
     * the stub progress function
     */
    ucs_queue_head_init(&tmp_pending_queue);
    ucs_queue_for_each_extract(uct_req, &stub_ep->pending_q, priv, 1) {
        ucs_queue_push(&tmp_pending_queue, ucp_stub_ep_req_priv(uct_req));
    }

    /* Switch to real transport */
    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        if (ep->uct_eps[lane] == &stub_ep->super) {
            ep->uct_eps[lane] = uct_ep;
            break;
        }
    }

    /* Destroy stub endpoint (destroys aux_ep as well) */
    uct_ep_destroy(&stub_ep->super);
    stub_ep = NULL;

    /* Replay pending requests */
    ucs_queue_for_each_extract(uct_req, &tmp_pending_queue, priv, 1) {
        req = ucs_container_of(uct_req, ucp_request_t, send.uct);
        ucs_assert(req->send.ep == ep);
        ucp_request_start_send(req);
        --ep->worker->stub_pend_count;
    }

    UCS_ASYNC_UNBLOCK(&ep->worker->async);
}

static ucs_status_t ucp_stub_ep_send_func(uct_ep_h uct_ep)
{
    return UCS_ERR_NO_RESOURCE;
}

static ssize_t ucp_stub_ep_bcopy_send_func(uct_ep_h uct_ep)
{
    return UCS_ERR_NO_RESOURCE;
}

static uct_ep_h ucp_stub_ep_get_wireup_msg_ep(ucp_stub_ep_t *stub_ep)
{
    uct_ep_h wireup_msg_ep;

    if (stub_ep->flags & UCP_STUB_EP_FLAG_READY) {
        wireup_msg_ep = stub_ep->next_ep;
    } else {
        wireup_msg_ep = stub_ep->aux_ep;
    }
    ucs_assertv(wireup_msg_ep != NULL,
                "ep=%p stub_ep=%p flags=%c%c next_ep=%p aux_ep=%p", stub_ep->ep,
                stub_ep,
                (stub_ep->flags & UCP_STUB_EP_FLAG_LOCAL_CONNECTED) ? 'c' : '-',
                (stub_ep->flags & UCP_STUB_EP_FLAG_READY)           ? 'r' : '-',
                stub_ep->next_ep, stub_ep->aux_ep);
    return wireup_msg_ep;
}

static ucs_status_t ucp_stub_ep_progress_pending(uct_pending_req_t *self)
{
    ucp_request_t *proxy_req = ucs_container_of(self, ucp_request_t, send.uct);
    uct_pending_req_t *req = proxy_req->send.proxy.req;
    ucp_stub_ep_t *stub_ep = proxy_req->send.proxy.stub_ep;
    ucs_status_t status;

    status = req->func(req);
    if (status == UCS_OK) {
        ucs_atomic_add32(&stub_ep->pending_count, -1);
        ucp_request_put(proxy_req);
    }
    return status;
}

static void ucp_stub_ep_pending_req_release(uct_pending_req_t *self,
                                            void *arg)
{
    ucp_request_t *proxy_req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_stub_ep_t *stub_ep = proxy_req->send.proxy.stub_ep;
    ucp_stub_ep_pending_release_proxy_arg_t *parg = arg;

    parg->cb(proxy_req->send.proxy.req, parg->arg);
    ucs_atomic_add32(&stub_ep->pending_count, -1);
    ucp_request_put(proxy_req);
}

static ucs_status_t ucp_stub_pending_add(uct_ep_h uct_ep, uct_pending_req_t *req)
{
    ucp_stub_ep_t *stub_ep = ucs_derived_of(uct_ep, ucp_stub_ep_t);
    ucp_ep_h ep = stub_ep->ep;
    ucp_worker_h worker = ep->worker;
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

        wireup_msg_ep                 = ucp_stub_ep_get_wireup_msg_ep(stub_ep);
        proxy_req->send.uct.func      = ucp_stub_ep_progress_pending;
        proxy_req->send.proxy.req     = req;
        proxy_req->send.proxy.stub_ep = stub_ep;

        status = uct_ep_pending_add(wireup_msg_ep, &proxy_req->send.uct);
        if (status == UCS_OK) {
            ucs_atomic_add32(&stub_ep->pending_count, +1);
        } else {
            ucp_request_put(proxy_req);
        }
    } else {
        ucs_queue_push(&stub_ep->pending_q, ucp_stub_ep_req_priv(req));
        ++ep->worker->stub_pend_count;
        status = UCS_OK;
    }
out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return status;
}

static void ucp_stub_pending_purge(uct_ep_h uct_ep,
                                   uct_pending_purge_callback_t cb,
                                   void *arg)
{
    ucp_stub_ep_t                           *stub_ep;
    ucp_stub_ep_pending_release_proxy_arg_t parg;
    uct_pending_req_t                       *req;
    ucp_request_t                           *ucp_req;

    stub_ep = ucs_derived_of(uct_ep, ucp_stub_ep_t);

    /* stub ep pending queue can be nonempty only on failure */
    ucs_assert_always(ucs_queue_is_empty(&stub_ep->pending_q) ||
                      UCS_PTR_IS_ERR(arg));

    if (stub_ep->aux_ep != NULL) {
        ucs_queue_for_each_extract(req, &stub_ep->pending_q, priv, 1) {
            ucp_req = ucs_container_of(req, ucp_request_t, send.uct);
            ucs_assert_always(UCS_PTR_IS_ERR(arg) &&
                              (cb == ucp_ep_err_pending_purge));
            ucp_request_complete_send(ucp_req, UCS_PTR_STATUS(arg));
        }

        parg.cb  = cb;
        parg.arg = arg;
        uct_ep_pending_purge(stub_ep->aux_ep, ucp_stub_ep_pending_req_release,
                             &parg);
    }
}

static ssize_t ucp_stub_ep_am_bcopy(uct_ep_h uct_ep, uint8_t id,
                                    uct_pack_callback_t pack_cb, void *arg,
                                    unsigned flags)
{
    ucp_stub_ep_t *stub_ep = ucs_derived_of(uct_ep, ucp_stub_ep_t);

    if (id == UCP_AM_ID_WIREUP) {
        return uct_ep_am_bcopy(ucp_stub_ep_get_wireup_msg_ep(stub_ep),
                               UCP_AM_ID_WIREUP, pack_cb, arg, flags);
    }

    return UCS_ERR_NO_RESOURCE;
}

static uct_iface_t ucp_stub_iface = {
    .ops = {
        .ep_get_address       = ucp_stub_ep_get_address,
        .ep_connect_to_ep     = ucp_stub_ep_connect_to_ep,
        .ep_flush             = (void*)ucs_empty_function_return_no_resource,
        .ep_destroy           = UCS_CLASS_DELETE_FUNC_NAME(ucp_stub_ep_t),
        .ep_pending_add       = ucp_stub_pending_add,
        .ep_pending_purge     = ucp_stub_pending_purge,
        .ep_put_short         = (void*)ucp_stub_ep_send_func,
        .ep_put_bcopy         = (void*)ucp_stub_ep_bcopy_send_func,
        .ep_put_zcopy         = (void*)ucp_stub_ep_send_func,
        .ep_get_bcopy         = (void*)ucp_stub_ep_send_func,
        .ep_get_zcopy         = (void*)ucp_stub_ep_send_func,
        .ep_am_short          = (void*)ucp_stub_ep_send_func,
        .ep_am_bcopy          = ucp_stub_ep_am_bcopy,
        .ep_am_zcopy          = (void*)ucp_stub_ep_send_func,
        .ep_tag_eager_short   = (void*)ucp_stub_ep_send_func,
        .ep_tag_eager_bcopy   = (void*)ucp_stub_ep_bcopy_send_func,
        .ep_tag_eager_zcopy   = (void*)ucp_stub_ep_send_func,
        .ep_tag_rndv_zcopy    = (void*)ucs_empty_function_return_ptr_no_resource,
        .ep_tag_rndv_request  = (void*)ucp_stub_ep_send_func,
        .ep_atomic_add64      = (void*)ucp_stub_ep_send_func,
        .ep_atomic_fadd64     = (void*)ucp_stub_ep_send_func,
        .ep_atomic_swap64     = (void*)ucp_stub_ep_send_func,
        .ep_atomic_cswap64    = (void*)ucp_stub_ep_send_func,
        .ep_atomic_add32      = (void*)ucp_stub_ep_send_func,
        .ep_atomic_fadd32     = (void*)ucp_stub_ep_send_func,
        .ep_atomic_swap32     = (void*)ucp_stub_ep_send_func,
        .ep_atomic_cswap32    = (void*)ucp_stub_ep_send_func,
        .ep_mem_reg_nc        = (void*)ucp_stub_ep_send_func,
    }
};

UCS_CLASS_DEFINE_NAMED_NEW_FUNC(ucp_stub_ep_create, ucp_stub_ep_t, uct_ep_t,
                                ucp_ep_h);

static ucs_status_t
ucp_stub_ep_connect_aux(ucp_stub_ep_t *stub_ep, unsigned address_count,
                        const ucp_address_entry_t *address_list)
{
    ucp_ep_h ep            = stub_ep->ep;
    ucp_worker_h worker    = ep->worker;
    const ucp_address_entry_t *aux_addr;
    unsigned aux_addr_index;
    ucs_status_t status;

    /* select an auxiliary transport which would be used to pass connection
     * establishment messages.
     */
    status = ucp_wireup_select_aux_transport(ep, address_list, address_count,
                                             &stub_ep->aux_rsc_index,
                                             &aux_addr_index);
    if (status != UCS_OK) {
        return status;
    }

    aux_addr = &address_list[aux_addr_index];

    /* create auxiliary endpoint connected to the remote iface. */
    status = uct_ep_create_connected(worker->ifaces[stub_ep->aux_rsc_index].iface,
                                     aux_addr->dev_addr, aux_addr->iface_addr,
                                     &stub_ep->aux_ep);
    if (status != UCS_OK) {
        return status;
    }

    ucs_debug("ep %p: stub_ep %p created aux_ep %p to %s using " UCT_TL_RESOURCE_DESC_FMT,
              ep, stub_ep, stub_ep->aux_ep, ucp_ep_peer_name(ep),
              UCT_TL_RESOURCE_DESC_ARG(&worker->context->tl_rscs[aux_addr_index].tl_rsc));
    return UCS_OK;
}

UCS_CLASS_INIT_FUNC(ucp_stub_ep_t, ucp_ep_h ep)
{
    self->super.iface   = &ucp_stub_iface;
    self->ep            = ep;
    self->aux_ep        = NULL;
    self->next_ep       = NULL;
    self->aux_rsc_index = UCP_NULL_RESOURCE;
    self->pending_count = 0;
    self->flags         = 0;
    self->progress_id   = UCS_CALLBACKQ_ID_NULL;
    ucs_queue_head_init(&self->pending_q);
    ucs_trace("ep %p: created stub ep %p to %s ", ep, self, ucp_ep_peer_name(ep));
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(ucp_stub_ep_t)
{
    ucs_assert(ucs_queue_is_empty(&self->pending_q));
    ucs_assert(self->pending_count == 0);

    ucs_debug("ep %p: destroy stub ep %p", self->ep, self);

    uct_worker_progress_unregister_safe(self->ep->worker->uct, &self->progress_id);
    if (self->aux_ep != NULL) {
        uct_ep_destroy(self->aux_ep);
    }
    if (self->next_ep != NULL) {
        uct_ep_destroy(self->next_ep);
    }
}

UCS_CLASS_DEFINE(ucp_stub_ep_t, void);

ucp_rsc_index_t ucp_stub_ep_get_aux_rsc_index(uct_ep_h uct_ep)
{
    ucp_stub_ep_t *stub_ep = ucs_derived_of(uct_ep, ucp_stub_ep_t);

    if (!ucp_stub_ep_test(uct_ep)) {
        return UCP_NULL_RESOURCE;
    }

    ucs_assert(stub_ep->aux_ep != NULL);
    return stub_ep->aux_rsc_index;
}

ucs_status_t ucp_stub_ep_connect(uct_ep_h uct_ep, ucp_rsc_index_t rsc_index,
                                 int connect_aux, unsigned address_count,
                                 const ucp_address_entry_t *address_list)
{
    ucp_stub_ep_t *stub_ep = ucs_derived_of(uct_ep, ucp_stub_ep_t);
    ucp_ep_h ep            = stub_ep->ep;
    ucp_worker_h worker    = ep->worker;
    ucs_status_t status;

    ucs_assert(ucp_stub_ep_test(uct_ep));

    status = uct_ep_create(worker->ifaces[rsc_index].iface, &stub_ep->next_ep);
    if (status != UCS_OK) {
        goto err;
    }

    ucs_debug("ep %p: created next_ep %p to %s using " UCT_TL_RESOURCE_DESC_FMT,
              ep, stub_ep->next_ep, ucp_ep_peer_name(ep),
              UCT_TL_RESOURCE_DESC_ARG(&worker->context->tl_rscs[rsc_index].tl_rsc));

    /* we need to create an auxiliary transport only for active messages */
    if (connect_aux) {
        status = ucp_stub_ep_connect_aux(stub_ep, address_count, address_list);
        if (status != UCS_OK) {
            goto err_destroy_next_ep;
        }
    }

    return UCS_OK;

err_destroy_next_ep:
    uct_ep_destroy(stub_ep->next_ep);
    stub_ep->next_ep = NULL;
err:
    return status;
}

void ucp_stub_ep_set_next_ep(uct_ep_h uct_ep, uct_ep_h next_ep)
{
    ucp_stub_ep_t *stub_ep = ucs_derived_of(uct_ep, ucp_stub_ep_t);

    ucs_assert(ucp_stub_ep_test(uct_ep));
    ucs_assert(stub_ep->next_ep == NULL);
    stub_ep->flags |= UCP_STUB_EP_FLAG_LOCAL_CONNECTED;
    stub_ep->next_ep = next_ep;
}

void ucp_stub_ep_remote_connected(uct_ep_h uct_ep)
{
    ucp_stub_ep_t *stub_ep = ucs_derived_of(uct_ep, ucp_stub_ep_t);

    ucs_assert(ucp_stub_ep_test(uct_ep));
    ucs_assert(stub_ep->next_ep != NULL);
    ucs_assert(stub_ep->flags & UCP_STUB_EP_FLAG_LOCAL_CONNECTED);

    ucs_trace("ep %p: stub ep %p is remote-connected", stub_ep->ep, stub_ep);
    stub_ep->flags |= UCP_STUB_EP_FLAG_READY;
    uct_worker_progress_register_safe(stub_ep->ep->worker->uct,
                                      ucp_stub_ep_progress, stub_ep, 0,
                                      &stub_ep->progress_id);
}

int ucp_stub_ep_test(uct_ep_h uct_ep)
{
    return uct_ep->iface == &ucp_stub_iface;
}

int ucp_stub_ep_test_aux(uct_ep_h stub_ep, uct_ep_h aux_ep)
{
    return ucp_stub_ep_test(stub_ep) &&
           (ucs_derived_of(stub_ep, ucp_stub_ep_t)->aux_ep == aux_ep);
}

uct_ep_h ucp_stub_ep_extract_aux(uct_ep_h ep)
{
    ucp_stub_ep_t *stub_ep = ucs_derived_of(ep, ucp_stub_ep_t);
    uct_ep_h       aux_ep  = stub_ep->aux_ep;

    stub_ep->aux_ep = NULL;
    return aux_ep;
}
