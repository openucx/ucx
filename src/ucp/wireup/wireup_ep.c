/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2019. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "wireup_ep.h"
#include "wireup.h"

#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_worker.h>
#include <ucs/arch/atomic.h>
#include <ucs/datastruct/queue.h>
#include <ucs/type/class.h>
#include <ucs/sys/sock.h>
#include <ucs/sys/stubs.h>
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
    ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(uct_ep);

    if (wireup_ep->flags & UCP_WIREUP_EP_FLAG_LOCAL_CONNECTED) {
        return UCS_OK;
    }

    wireup_ep->flags |= UCP_WIREUP_EP_FLAG_LOCAL_CONNECTED;
    return uct_ep_connect_to_ep(wireup_ep->super.uct_ep, dev_addr, ep_addr);
}

static ssize_t ucp_wireup_ep_bcopy_send_func(uct_ep_h uct_ep)
{
    return UCS_ERR_NO_RESOURCE;
}

static int ucp_wireup_ep_is_next_ep_active(ucp_wireup_ep_t *wireup_ep)
{
    return (wireup_ep->flags & UCP_WIREUP_EP_FLAG_READY) ||
           (wireup_ep->aux_ep == NULL);
}

uct_ep_h ucp_wireup_ep_extract_msg_ep(ucp_wireup_ep_t *wireup_ep)
{
    uct_ep_h msg_ep;

    ucs_assertv(ucs_queue_is_empty(&wireup_ep->pending_q), "queue_length=%zu",
                ucs_queue_length(&wireup_ep->pending_q));
    uct_ep_pending_purge(&wireup_ep->super.super, NULL, NULL);

    if (ucp_wireup_ep_is_next_ep_active(wireup_ep)) {
        return ucp_wireup_ep_extract_next_ep(&wireup_ep->super.super);
    }

    msg_ep                   = wireup_ep->aux_ep;
    wireup_ep->aux_ep        = NULL;
    wireup_ep->aux_rsc_index = UCP_NULL_RESOURCE;
    return msg_ep;
}

uct_ep_h ucp_wireup_ep_get_msg_ep(ucp_wireup_ep_t *wireup_ep)
{
    uct_ep_h wireup_msg_ep;

    if (ucp_wireup_ep_is_next_ep_active(wireup_ep)) {
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

ucs_status_t ucp_wireup_ep_progress_pending(uct_pending_req_t *self)
{
    ucp_request_t *proxy_req = ucs_container_of(self, ucp_request_t, send.uct);
    uct_pending_req_t *req = proxy_req->send.proxy.req;
    ucp_wireup_ep_t *wireup_ep = proxy_req->send.proxy.wireup_ep;
    ucs_status_t status;

    status = req->func(req);
    if (status == UCS_OK) {
        ucs_atomic_sub32(&wireup_ep->pending_count, 1);
        ucp_request_mem_free(proxy_req);
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

    ucs_atomic_sub32(&wireup_ep->pending_count, 1);

    if (proxy_req->send.proxy.req->func == ucp_wireup_msg_progress) {
        req = ucs_container_of(proxy_req->send.proxy.req, ucp_request_t,
                               send.uct);
        ucs_free(req->send.buffer);
        ucp_request_mem_free(req);
    }

    ucs_free(proxy_req);
}

static ucs_status_t ucp_wireup_ep_pending_add(uct_ep_h uct_ep,
                                              uct_pending_req_t *req,
                                              unsigned flags)
{
    ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(uct_ep);
    ucp_ep_h ucp_ep = wireup_ep->super.ucp_ep;
    ucp_worker_h worker = ucp_ep->worker;
    ucp_request_t *proxy_req;
    uct_ep_h wireup_msg_ep;
    ucs_status_t status;

    UCS_ASYNC_BLOCK(&worker->async);
    if (req->func == ucp_wireup_msg_progress) {
        proxy_req = ucp_request_mem_alloc("ucp_wireup_proxy_req");
        if (proxy_req == NULL) {
            status = UCS_ERR_NO_MEMORY;
            goto out;
        }

        wireup_msg_ep = ucp_wireup_ep_get_msg_ep(wireup_ep);

        proxy_req->send.uct.func            = ucp_wireup_ep_progress_pending;
        proxy_req->send.proxy.req           = req;
        proxy_req->send.proxy.wireup_ep     = wireup_ep;
        proxy_req->send.state.uct_comp.func = NULL;

        status = uct_ep_pending_add(wireup_msg_ep, &proxy_req->send.uct,
                                    UCT_CB_FLAG_ASYNC);
        if (status == UCS_OK) {
            ucs_atomic_add32(&wireup_ep->pending_count, +1);
        } else {
            ucp_request_mem_free(proxy_req);
        }
    } else {
        ucs_queue_push(&wireup_ep->pending_q, ucp_wireup_ep_req_priv(req));
        ucp_worker_flush_ops_count_add(worker, +1);
        status = UCS_OK;
    }
out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    /* coverity[leaked_storage] */
    return status;
}

void ucp_wireup_ep_pending_queue_purge(uct_ep_h uct_ep,
                                       uct_pending_purge_callback_t cb,
                                       void *arg)
{
    ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(uct_ep);
    ucp_worker_h worker        = wireup_ep->super.ucp_ep->worker;
    uct_pending_req_t *req;
    ucp_request_t *ucp_req;

    ucs_queue_for_each_extract(req, &wireup_ep->pending_q, priv, 1) {
        ucp_req = ucs_container_of(req, ucp_request_t, send.uct);
        UCS_ASYNC_BLOCK(&worker->async);
        ucp_worker_flush_ops_count_add(worker, -1);
        UCS_ASYNC_UNBLOCK(&worker->async);
        cb(&ucp_req->send.uct, arg);
    }
}

static void
ucp_wireup_ep_pending_purge(uct_ep_h uct_ep, uct_pending_purge_callback_t cb,
                            void *arg)
{
    ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(uct_ep);

    ucp_wireup_ep_pending_queue_purge(uct_ep, cb, arg);

    if (wireup_ep->pending_count > 0) {
        uct_ep_pending_purge(ucp_wireup_ep_get_msg_ep(wireup_ep),
                             ucp_wireup_ep_pending_req_release, arg);
    }

    ucs_assert(wireup_ep->pending_count == 0);
}

static ssize_t ucp_wireup_ep_am_bcopy(uct_ep_h uct_ep, uint8_t id,
                                      uct_pack_callback_t pack_cb, void *arg,
                                      unsigned flags)
{
    ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(uct_ep);

    if (id == UCP_AM_ID_WIREUP) {
        return uct_ep_am_bcopy(ucp_wireup_ep_get_msg_ep(wireup_ep),
                               UCP_AM_ID_WIREUP, pack_cb, arg, flags);
    }

    return UCS_ERR_NO_RESOURCE;
}


UCS_CLASS_DEFINE_NAMED_NEW_FUNC(ucp_wireup_ep_create, ucp_wireup_ep_t, uct_ep_t,
                                ucp_ep_h);

void ucp_wireup_ep_set_aux(ucp_wireup_ep_t *wireup_ep, uct_ep_h uct_ep,
                           ucp_rsc_index_t rsc_index, int is_p2p)
{
    ucp_worker_iface_t *wiface =
        ucp_worker_iface(wireup_ep->super.ucp_ep->worker, rsc_index);

    ucs_assert(!ucp_wireup_ep_test(uct_ep));
    wireup_ep->aux_ep        = uct_ep;
    wireup_ep->aux_rsc_index = rsc_index;

    if (is_p2p) {
        wireup_ep->flags |= UCP_WIREUP_EP_FLAG_AUX_P2P;
    }

    ucp_worker_iface_progress_ep(wiface);
}

static ucs_status_t
ucp_wireup_ep_connect_aux(ucp_wireup_ep_t *wireup_ep, unsigned ep_init_flags,
                          const ucp_unpacked_address_t *remote_address)
{
    ucp_ep_h ucp_ep                      = wireup_ep->super.ucp_ep;
    ucp_worker_h worker                  = ucp_ep->worker;
    ucp_wireup_select_info_t select_info = {0};
    uct_ep_params_t uct_ep_params;
    const ucp_address_entry_t *aux_addr;
    ucp_worker_iface_t *wiface;
    ucs_status_t status;
    uct_ep_h uct_ep;

    /* select an auxiliary transport which would be used to pass connection
     * establishment messages.
     */
    status = ucp_wireup_select_aux_transport(ucp_ep, ep_init_flags,
                                             ucp_tl_bitmap_max, remote_address,
                                             &select_info);
    if (status != UCS_OK) {
        return status;
    }

    aux_addr = &remote_address->address_list[select_info.addr_index];
    wiface   = ucp_worker_iface(worker, select_info.rsc_index);

    /* create auxiliary endpoint connected to the remote iface. */
    uct_ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE    |
                               UCT_EP_PARAM_FIELD_DEV_ADDR |
                               UCT_EP_PARAM_FIELD_IFACE_ADDR;
    uct_ep_params.iface      = wiface->iface;
    uct_ep_params.dev_addr   = aux_addr->dev_addr;
    uct_ep_params.iface_addr = aux_addr->iface_addr;
    status = uct_ep_create(&uct_ep_params, &uct_ep);
    if (status != UCS_OK) {
        /* coverity[leaked_storage] */
        return status;
    }

    ucp_wireup_ep_set_aux(wireup_ep, uct_ep, select_info.rsc_index, 0);

    ucs_debug("ep %p: wireup_ep %p created aux_ep %p to %s using "
              UCT_TL_RESOURCE_DESC_FMT, ucp_ep, wireup_ep, wireup_ep->aux_ep,
              ucp_ep_peer_name(ucp_ep),
              UCT_TL_RESOURCE_DESC_ARG(&worker->context->tl_rscs[select_info.rsc_index].tl_rsc));

    return UCS_OK;
}

void ucp_wireup_ep_discard_aux_ep(ucp_wireup_ep_t *wireup_ep,
                                  unsigned ep_flush_flags,
                                  uct_pending_purge_callback_t purge_cb,
                                  void *purge_arg)
{
    ucp_ep_h ucp_ep     = wireup_ep->super.ucp_ep;
    uct_ep_h aux_ep     = wireup_ep->aux_ep;
    ucp_worker_h worker = ucp_ep->worker;
    ucp_rsc_index_t rsc_index;

    if (aux_ep == NULL) {
        return;
    }

    ucp_wireup_ep_disown(&wireup_ep->super.super, aux_ep);
    rsc_index = wireup_ep->aux_rsc_index;
    ucp_worker_discard_uct_ep(ucp_ep, aux_ep, rsc_index, ep_flush_flags,
                              purge_cb, purge_arg,
                              (ucp_send_nbx_callback_t)ucs_empty_function,
                              NULL);
    if (worker->context->config.ext.proto_enable) {
        ucp_worker_iface_unprogress_ep(ucp_worker_iface(worker, rsc_index));
    }
}

static ucs_status_t ucp_wireup_ep_flush(uct_ep_h uct_ep, unsigned flags,
                                        uct_completion_t *comp)
{
    ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(uct_ep);

    if (flags & UCT_FLUSH_FLAG_CANCEL) {
        if (wireup_ep->aux_ep) {
            return uct_ep_flush(wireup_ep->aux_ep, flags, comp);
        }
        return UCS_OK;
    }
    return UCS_ERR_NO_RESOURCE;
}

static ucs_status_t ucp_wireup_ep_do_check(ucp_ep_h ucp_ep, uct_ep_h uct_ep,
                                           ucp_rsc_index_t rsc_idx, int is_p2p,
                                           unsigned flags,
                                           uct_completion_t *comp)
{
    ucp_worker_h worker = ucp_ep->worker;
    ucp_worker_iface_t *wiface;

    ucs_assert(rsc_idx != UCP_NULL_RESOURCE);

    wiface = ucp_worker_iface(worker, rsc_idx);
    if (ucp_ep_is_am_keepalive(ucp_ep, rsc_idx, is_p2p)) {
        return ucp_ep_do_uct_ep_am_keepalive(ucp_ep, uct_ep, rsc_idx);
    } else if (wiface->attr.cap.flags & UCT_IFACE_FLAG_EP_CHECK) {
        return uct_ep_check(uct_ep, flags, comp);
    }

    /* If EP_CHECK is not supported by UCT transport and AM-based keepalive is
     * not required, it has to support a built-in keepalive mechanism to be
     * able to detect peer failure during wireup
     */
    ucs_assert(wiface->attr.cap.flags & UCT_IFACE_FLAG_EP_KEEPALIVE);
    return UCS_OK;
}

static ucs_status_t ucp_wireup_ep_check(uct_ep_h uct_ep, unsigned flags,
                                        uct_completion_t *comp)
{
    ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(uct_ep);
    ucp_ep_h ucp_ep            = wireup_ep->super.ucp_ep;

    if (wireup_ep->flags & UCP_WIREUP_EP_FLAG_REMOTE_CONNECTED) {
        return uct_ep_check(wireup_ep->super.uct_ep, flags, comp);
    }

    if (wireup_ep->aux_ep != NULL) {
        return ucp_wireup_ep_do_check(ucp_ep, wireup_ep->aux_ep,
                                      wireup_ep->aux_rsc_index,
                                      wireup_ep->flags &
                                              UCP_WIREUP_EP_FLAG_AUX_P2P,
                                      flags, comp);
    }

    ucs_trace("ep %p: wireup_ep %p skipping keepalive, flags 0x%x", ucp_ep,
              wireup_ep, wireup_ep->flags);
    return UCS_OK;
}


UCS_CLASS_INIT_FUNC(ucp_wireup_ep_t, ucp_ep_h ucp_ep)
{
    static uct_iface_ops_t ops = {
        .ep_connect_to_ep    = ucp_wireup_ep_connect_to_ep,
        .ep_flush            = ucp_wireup_ep_flush,
        .ep_check            = ucp_wireup_ep_check,
        .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(ucp_wireup_ep_t),
        .ep_pending_add      = ucp_wireup_ep_pending_add,
        .ep_pending_purge    = ucp_wireup_ep_pending_purge,
        .ep_put_short        = (uct_ep_put_short_func_t)ucs_empty_function_return_no_resource,
        .ep_put_bcopy        = (uct_ep_put_bcopy_func_t)ucp_wireup_ep_bcopy_send_func,
        .ep_put_zcopy        = (uct_ep_put_zcopy_func_t)ucs_empty_function_return_no_resource,
        .ep_get_short        = (uct_ep_get_short_func_t)ucs_empty_function_return_no_resource,
        .ep_get_bcopy        = (uct_ep_get_bcopy_func_t)ucs_empty_function_return_no_resource,
        .ep_get_zcopy        = (uct_ep_get_zcopy_func_t)ucs_empty_function_return_no_resource,
        .ep_am_short         = (uct_ep_am_short_func_t)ucs_empty_function_return_no_resource,
        .ep_am_short_iov     = (uct_ep_am_short_iov_func_t)ucs_empty_function_return_no_resource,
        .ep_am_bcopy         = ucp_wireup_ep_am_bcopy,
        .ep_am_zcopy         = (uct_ep_am_zcopy_func_t)ucs_empty_function_return_no_resource,
        .ep_tag_eager_short  = (uct_ep_tag_eager_short_func_t)ucs_empty_function_return_no_resource,
        .ep_tag_eager_bcopy  = (uct_ep_tag_eager_bcopy_func_t)ucp_wireup_ep_bcopy_send_func,
        .ep_tag_eager_zcopy  = (uct_ep_tag_eager_zcopy_func_t)ucs_empty_function_return_no_resource,
        .ep_tag_rndv_zcopy   = (uct_ep_tag_rndv_zcopy_func_t)ucs_empty_function_return_ptr_no_resource,
        .ep_tag_rndv_request = (uct_ep_tag_rndv_request_func_t)ucs_empty_function_return_no_resource,
        .ep_atomic64_post    = (uct_ep_atomic64_post_func_t)ucs_empty_function_return_no_resource,
        .ep_atomic64_fetch   = (uct_ep_atomic64_fetch_func_t)ucs_empty_function_return_no_resource,
        .ep_atomic_cswap64   = (uct_ep_atomic_cswap64_func_t)ucs_empty_function_return_no_resource,
        .ep_atomic32_post    = (uct_ep_atomic32_post_func_t)ucs_empty_function_return_no_resource,
        .ep_atomic32_fetch   = (uct_ep_atomic32_fetch_func_t)ucs_empty_function_return_no_resource,
        .ep_atomic_cswap32   = (uct_ep_atomic_cswap32_func_t)ucs_empty_function_return_no_resource
    };

    UCS_CLASS_CALL_SUPER_INIT(ucp_proxy_ep_t, &ops, ucp_ep, NULL, 0);

    self->aux_ep        = NULL;
    self->aux_rsc_index = UCP_NULL_RESOURCE;
    self->pending_count = 0;
    self->flags         = 0;
    ucs_queue_head_init(&self->pending_q);
    UCS_STATIC_BITMAP_RESET_ALL(&self->cm_resolve_tl_bitmap);

    UCS_ASYNC_BLOCK(&ucp_ep->worker->async);
    ucp_worker_flush_ops_count_add(ucp_ep->worker, +1);
    UCS_ASYNC_UNBLOCK(&ucp_ep->worker->async);

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

    if (self->aux_ep != NULL) {
        /* No pending operations should be scheduled */
        ucp_wireup_ep_discard_aux_ep(self, UCT_FLUSH_FLAG_CANCEL,
                                     ucp_destroyed_ep_pending_purge, ucp_ep);
        self->aux_ep = NULL;
    }

    if (self->super.is_owner && (self->super.uct_ep != NULL)) {
        /* No pending operations should be scheduled */
        ucp_worker_discard_uct_ep(self->super.ucp_ep, self->super.uct_ep,
                                  self->super.rsc_index, UCT_FLUSH_FLAG_CANCEL,
                                  ucp_destroyed_ep_pending_purge, ucp_ep,
                                  (ucp_send_nbx_callback_t)ucs_empty_function,
                                  NULL);
        ucp_proxy_ep_set_uct_ep(&self->super, NULL, 0, UCP_NULL_RESOURCE);
    }

    UCS_ASYNC_BLOCK(&worker->async);
    ucp_worker_flush_ops_count_add(worker, -1);
    UCS_ASYNC_UNBLOCK(&worker->async);
}

UCS_CLASS_DEFINE(ucp_wireup_ep_t, ucp_proxy_ep_t);

ucp_rsc_index_t ucp_wireup_ep_get_aux_rsc_index(uct_ep_h uct_ep)
{
    ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(uct_ep);

    if (!ucp_wireup_ep_test(uct_ep)) {
        return UCP_NULL_RESOURCE;
    }

    if (wireup_ep->aux_ep == NULL) {
        return UCP_NULL_RESOURCE;
    }

    return wireup_ep->aux_rsc_index;
}

ucs_status_t ucp_wireup_ep_connect(uct_ep_h uct_ep, unsigned ep_init_flags,
                                   ucp_rsc_index_t rsc_index,
                                   unsigned path_index, int connect_aux,
                                   const ucp_unpacked_address_t *remote_address)
{
    ucp_wireup_ep_t *wireup_ep     = ucp_wireup_ep(uct_ep);
    ucp_ep_h ucp_ep                = wireup_ep->super.ucp_ep;
    ucp_worker_h worker            = ucp_ep->worker;
    uct_ep_params_t uct_ep_params;
    ucs_status_t status;
    uct_ep_h next_ep;

    ucs_assert(wireup_ep != NULL);

    uct_ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE |
                               UCT_EP_PARAM_FIELD_PATH_INDEX;
    uct_ep_params.path_index = path_index;
    uct_ep_params.iface      = ucp_worker_iface(worker, rsc_index)->iface;
    status = uct_ep_create(&uct_ep_params, &next_ep);
    if (status != UCS_OK) {
        /* make Coverity happy */
        ucs_assert(next_ep == NULL);
        goto err;
    }

    ucp_proxy_ep_set_uct_ep(&wireup_ep->super, next_ep, 1, rsc_index);

    ucs_debug("ep %p: wireup_ep %p created next_ep %p to %s "
              "using " UCT_TL_RESOURCE_DESC_FMT,
              ucp_ep, wireup_ep, wireup_ep->super.uct_ep,
              ucp_ep_peer_name(ucp_ep),
              UCT_TL_RESOURCE_DESC_ARG(
                      &worker->context->tl_rscs[rsc_index].tl_rsc));

    /* We need to create an auxiliary transport only for active messages.
       Skip this step if auxiliary already exists. */
    if (connect_aux && (wireup_ep->aux_ep == NULL)) {
        status = ucp_wireup_ep_connect_aux(wireup_ep, ep_init_flags,
                                           remote_address);
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

int ucp_wireup_ep_has_next_ep(ucp_wireup_ep_t *wireup_ep)
{
    ucs_assert(wireup_ep != NULL);
    return wireup_ep->super.uct_ep != NULL;
}

void ucp_wireup_ep_set_next_ep(uct_ep_h uct_ep, uct_ep_h next_ep,
                               ucp_rsc_index_t rsc_index)
{
    ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(uct_ep);

    ucs_assert(wireup_ep != NULL);
    ucs_assert(wireup_ep->super.uct_ep == NULL);
    ucs_assert(!ucp_wireup_ep_test(next_ep));
    wireup_ep->flags |= UCP_WIREUP_EP_FLAG_LOCAL_CONNECTED;
    ucp_proxy_ep_set_uct_ep(&wireup_ep->super, next_ep, 1, rsc_index);
    ucs_debug("ep %p: wireup_ep %p set next_ep %p", wireup_ep->super.ucp_ep,
              wireup_ep, wireup_ep->super.uct_ep);
}

uct_ep_h ucp_wireup_ep_extract_next_ep(uct_ep_h uct_ep)
{
    ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(uct_ep);
    uct_ep_h next_ep;

    ucs_assert_always(wireup_ep != NULL);
    next_ep = wireup_ep->super.uct_ep;
    ucp_proxy_ep_set_uct_ep(&wireup_ep->super, NULL, 0, UCP_NULL_RESOURCE);
    return next_ep;
}

void ucp_wireup_ep_destroy_next_ep(ucp_wireup_ep_t *wireup_ep)
{
    uct_ep_h uct_ep;

    ucs_assert(wireup_ep != NULL);
    uct_ep = ucp_wireup_ep_extract_next_ep(&wireup_ep->super.super);
    ucs_assert_always(uct_ep != NULL);
    uct_ep_destroy(uct_ep);

    wireup_ep->flags &= ~UCP_WIREUP_EP_FLAG_LOCAL_CONNECTED;
    ucs_assert((wireup_ep->flags & ~UCP_WIREUP_EP_FLAG_SEND_CLIENT_ID) == 0);
}

int ucp_wireup_ep_test(uct_ep_h uct_ep)
{
    return uct_ep->iface->ops.ep_destroy ==
                    UCS_CLASS_DELETE_FUNC_NAME(ucp_wireup_ep_t);
}

int ucp_wireup_aux_ep_is_owner(ucp_wireup_ep_t *wireup_ep, uct_ep_h owned_ep)
{
    ucp_ep_h ucp_ep              = wireup_ep->super.ucp_ep;
    ucp_lane_index_t cm_lane_idx = ucp_ep_get_cm_lane(ucp_ep);

    return (wireup_ep->aux_ep == owned_ep) ||
           /* Auxiliary EP can be WIREUP EP in case of it is on CM lane */
           ((wireup_ep->aux_ep != NULL) && (cm_lane_idx != UCP_NULL_LANE) &&
            (ucp_ep_get_lane(ucp_ep, cm_lane_idx) == &wireup_ep->super.super) &&
            ucp_wireup_ep_is_owner(wireup_ep->aux_ep, owned_ep));
}

int ucp_wireup_ep_is_owner(uct_ep_h uct_ep, uct_ep_h owned_ep)
{
    ucp_wireup_ep_t *wireup_ep;

    if (uct_ep == NULL) {
        return 0;
    }

    wireup_ep = ucp_wireup_ep(uct_ep);
    if (wireup_ep == NULL) {
        return 0;
    }

    if ((ucp_wireup_aux_ep_is_owner(wireup_ep, owned_ep)) ||
        (wireup_ep->super.uct_ep == owned_ep)) {
        return 1;
    }

    return 0;
}

void ucp_wireup_ep_disown(uct_ep_h uct_ep, uct_ep_h owned_ep)
{
    ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(uct_ep);

    ucs_assert_always(wireup_ep != NULL);
    if (wireup_ep->aux_ep == owned_ep) {
        wireup_ep->aux_ep = NULL;
    } else if (wireup_ep->super.uct_ep == owned_ep) {
        ucp_proxy_ep_extract(uct_ep);
    }
}

ucp_wireup_ep_t *ucp_wireup_ep(uct_ep_h uct_ep)
{
    return ucp_wireup_ep_test(uct_ep) ?
           ucs_derived_of(uct_ep, ucp_wireup_ep_t) : NULL;
}

unsigned ucp_wireup_ep_pending_extract(ucp_wireup_ep_t *wireup_ep,
                                       ucs_queue_head_t *queue)
{
    unsigned count = 0;
    uct_pending_req_t *uct_req;

    ucs_queue_for_each_extract(uct_req, &wireup_ep->pending_q, priv, 1) {
        ucs_queue_push(queue, ucp_wireup_ep_req_priv(uct_req));
        ++count;
    }

    return count;
}

void ucp_wireup_eps_pending_extract(ucp_ep_t *ucp_ep, ucs_queue_head_t *queue)
{
    int pending_count = 0;
    ucp_lane_index_t lane_idx;
    ucp_wireup_ep_t *wireup_ep;
    uct_ep_h uct_ep;

    UCP_WORKER_THREAD_CS_CHECK_IS_BLOCKED(ucp_ep->worker);
    ucs_queue_head_init(queue);

    if (ucp_ep->cfg_index == UCP_WORKER_CFG_INDEX_NULL) {
        return;
    }

    for (lane_idx = 0; lane_idx < ucp_ep_num_lanes(ucp_ep); ++lane_idx) {
        uct_ep = ucp_ep_get_lane(ucp_ep, lane_idx);
        /* When creating EP with remote worker address
         * EP is using transport lanes only, with no CM lane. */
        if ((uct_ep == NULL) || (ucp_wireup_ep(uct_ep) == NULL)) {
            continue;
        }

        wireup_ep      = ucp_wireup_ep(uct_ep);
        pending_count += ucp_wireup_ep_pending_extract(wireup_ep, queue);
    }

    ucp_worker_flush_ops_count_add(ucp_ep->worker, -pending_count);
}

ucs_status_t
ucp_wireup_ep_connect_to_ep_v2(uct_ep_h tl_ep,
                               const ucp_address_entry_t *address_entry,
                               const ucp_address_entry_ep_addr_t *ep_entry)
{
    const uct_ep_connect_to_ep_params_t param = {
        .field_mask = UCT_EP_CONNECT_TO_EP_PARAM_FIELD_DEVICE_ADDR_LENGTH |
                      UCT_EP_CONNECT_TO_EP_PARAM_FIELD_EP_ADDR_LENGTH,
        .device_addr_length = address_entry->dev_addr_len,
        .ep_addr_length     = ep_entry->len
    };
    ucp_wireup_ep_t *wireup_ep                = ucp_wireup_ep(tl_ep);

    if (wireup_ep == NULL) {
        return uct_ep_connect_to_ep_v2(tl_ep, address_entry->dev_addr,
                                       ep_entry->addr, &param);
    }

    if (wireup_ep->flags & UCP_WIREUP_EP_FLAG_LOCAL_CONNECTED) {
        return UCS_OK;
    }

    wireup_ep->flags |= UCP_WIREUP_EP_FLAG_LOCAL_CONNECTED;
    return uct_ep_connect_to_ep_v2(wireup_ep->super.uct_ep,
                                   address_entry->dev_addr, ep_entry->addr,
                                   &param);
}
