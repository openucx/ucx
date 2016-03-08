/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "stub_ep.h"
#include "wireup.h"

#include <ucp/core/ucp_worker.h>
#include <ucs/datastruct/queue.h>


static UCS_CLASS_DEFINE_DELETE_FUNC(ucp_stub_ep_t, uct_ep_t);

static ucs_status_t ucp_stub_ep_send_func(uct_ep_h uct_ep)
{
    ucp_stub_ep_t *stub_ep = ucs_derived_of(uct_ep, ucp_stub_ep_t);
    ucp_wireup_progress(stub_ep->ep);
    return UCS_ERR_NO_RESOURCE;
}

static ssize_t ucp_stub_ep_bcopy_send_func(uct_ep_h uct_ep)
{
    ucp_stub_ep_t *stub_ep = ucs_derived_of(uct_ep, ucp_stub_ep_t);
    ucp_wireup_progress(stub_ep->ep);
    return UCS_ERR_NO_RESOURCE;
}

void ucp_stub_ep_progress(void *arg)
{
    ucp_stub_ep_t *dummy_ep = arg;
    ucp_wireup_progress(dummy_ep->ep);
}

static ucs_status_t ucp_stub_pending_add(uct_ep_h uct_ep, uct_pending_req_t *req)
{
    ucp_stub_ep_t *stub_ep = ucs_derived_of(uct_ep, ucp_stub_ep_t);
    ucp_ep_h ep = stub_ep->ep;

    ucs_queue_push(&stub_ep->pending_q, ucp_stub_ep_req_priv(req));
    ++ep->worker->stub_pend_count;

    /* Add a reference to the dummy progress function. If we have a pending
     * request and this endpoint is still doing wireup, we must make sure progress
     * is made. */
    uct_worker_progress_register(ep->worker->uct, ucp_stub_ep_progress, stub_ep);
    return UCS_OK;
}

static void ucp_stub_pending_purge(uct_ep_h uct_ep, uct_pending_callback_t cb)
{
    ucp_stub_ep_t *stub_ep = ucs_derived_of(uct_ep, ucp_stub_ep_t);
    ucs_assert_always(ucs_queue_is_empty(&stub_ep->pending_q));
}

UCS_CLASS_INIT_FUNC(ucp_stub_ep_t, ucp_ep_h ucp_ep) {

    memset(&self->iface, 0, sizeof(self->iface));
    self->iface.ops.ep_flush          = (void*)ucs_empty_function_return_inprogress;
    self->iface.ops.ep_destroy        = UCS_CLASS_DELETE_FUNC_NAME(ucp_stub_ep_t);
    self->iface.ops.ep_pending_add    = ucp_stub_pending_add;
    self->iface.ops.ep_pending_purge  = ucp_stub_pending_purge;
    self->iface.ops.ep_put_short      = (void*)ucp_stub_ep_send_func;
    self->iface.ops.ep_put_bcopy      = (void*)ucp_stub_ep_bcopy_send_func;
    self->iface.ops.ep_put_zcopy      = (void*)ucp_stub_ep_send_func;
    self->iface.ops.ep_get_bcopy      = (void*)ucp_stub_ep_send_func;
    self->iface.ops.ep_get_zcopy      = (void*)ucp_stub_ep_send_func;
    self->iface.ops.ep_am_short       = (void*)ucp_stub_ep_send_func;
    self->iface.ops.ep_am_bcopy       = (void*)ucp_stub_ep_bcopy_send_func;
    self->iface.ops.ep_am_zcopy       = (void*)ucp_stub_ep_send_func;
    self->iface.ops.ep_atomic_add64   = (void*)ucp_stub_ep_send_func;
    self->iface.ops.ep_atomic_fadd64  = (void*)ucp_stub_ep_send_func;
    self->iface.ops.ep_atomic_swap64  = (void*)ucp_stub_ep_send_func;
    self->iface.ops.ep_atomic_cswap64 = (void*)ucp_stub_ep_send_func;
    self->iface.ops.ep_atomic_add32   = (void*)ucp_stub_ep_send_func;
    self->iface.ops.ep_atomic_fadd32  = (void*)ucp_stub_ep_send_func;
    self->iface.ops.ep_atomic_swap32  = (void*)ucp_stub_ep_send_func;
    self->iface.ops.ep_atomic_cswap32 = (void*)ucp_stub_ep_send_func;

    self->super.iface                 = &self->iface;
    self->ep                          = ucp_ep;
    self->aux_ep                      = NULL;
    self->next_ep                     = NULL;
    self->pending_count               = 0;

    ucs_queue_head_init(&self->pending_q);

    return UCS_OK;
}

UCS_CLASS_CLEANUP_FUNC(ucp_stub_ep_t) {
}

UCS_CLASS_DEFINE(ucp_stub_ep_t, void);

ucp_stub_ep_t* ucp_ep_get_stub_ep(ucp_ep_h ep)
{
    ucs_assert(ep->state & UCP_EP_STATE_STUB_EP);
    return ucs_derived_of(ep->uct_ep, ucp_stub_ep_t);
}
