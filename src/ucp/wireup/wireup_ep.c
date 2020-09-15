/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
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

    ucs_assert(!(wireup_ep->flags & UCP_WIREUP_EP_FLAG_LOCAL_CONNECTED));
    wireup_ep->flags |= UCP_WIREUP_EP_FLAG_LOCAL_CONNECTED;
    return uct_ep_connect_to_ep(wireup_ep->super.uct_ep, dev_addr, ep_addr);
}

static void
ucp_wireup_tmp_ep_destroy_complete_cb(void *request, ucs_status_t status,
                                      void *user_data)
{
    ucp_wireup_ep_t *wireup_ep = (ucp_wireup_ep_t*)user_data;

    /* check for NULL pointer to workaround Coverity warning (it wrongly
     * assumes that this callback could be called upon GET/PUT operation) */
    ucs_assertv_always(wireup_ep != NULL,
                       "req=%p: user_data passed to the TMP EP destroy cb "
                       "mustn't be NULL", (ucp_request_t*)request - 1);

    wireup_ep->flags &= ~UCP_WIREUP_EP_FLAG_DESTROY_TMP_EP;
    ucs_assert(wireup_ep->tmp_ep == NULL);
    ucp_request_release(request);
}

static unsigned ucp_wireup_tmp_ep_disconnect_progress(void *arg)
{
    ucp_request_t *req         = (ucp_request_t*)arg;
    ucp_ep_h tmp_ep            = req->send.ep;
    ucp_worker_h worker        = tmp_ep->worker;
    ucs_async_context_t *async = &worker->async;

    UCS_ASYNC_BLOCK(async);
    ucp_ep_disconnected(tmp_ep, 1);
    --worker->flush_ops_count;
    ucp_request_complete_send(req, req->status);
    UCS_ASYNC_UNBLOCK(async);

    return 1;
}

static void ucp_wireup_tmp_ep_flushed_cb(ucp_request_t *req)
{
    uct_worker_cb_id_t cb_id = UCS_CALLBACKQ_ID_NULL;
    ucp_ep_h tmp_ep          = req->send.ep;

    /* schedule the destorying of TMP EP lanes to the main thread to
     * not destroy UCT EP from the UCT EP flush callback, since
     * UCT EP maybe touched during the progress after calling UCT
     * flush completion callback */
    uct_worker_progress_register_safe(tmp_ep->worker->uct,
                                      ucp_wireup_tmp_ep_disconnect_progress,
                                      req, UCS_CALLBACKQ_FLAG_ONESHOT, &cb_id);
}

/* the following values could be returned from the function:
 * - true:  destroying of the TMP EP was completed inplace, if the complete_cb
 *          was specified, it wouldn't be called
 * - false: destroying of the TMP EP is in progress now, if the complete_cb
 *          was specified, it would be called upon completion the destroying
 *          of the TMP EP */
int ucp_wireup_tmp_ep_destroy(ucp_ep_h ep, ucp_wireup_ep_t *wireup_ep,
                              unsigned ep_flush_flags,
                              ucp_send_nbx_callback_t complete_cb)
{
    ucp_ep_h tmp_ep           = wireup_ep->tmp_ep;
    ucp_worker_h worker       = tmp_ep->worker;
    ucp_request_param_t param = ucp_request_null_param;
    ucp_lane_index_t lane, found_lane;
    uct_ep_h uct_ep;
    void *req;

    ucs_assert(tmp_ep != ep);

    /* to prevent flush+destroy UCT EPs that are used by the main EP,
     * they have to be removed from the TMP EP lanes and their WIREUP
     * EPs have to be destroyed */
    for (lane = 0; lane < ucp_ep_num_lanes(tmp_ep); ++lane) {
        if (tmp_ep->uct_eps[lane] != NULL) {
            found_lane =
                ucp_wireup_ep_configs_can_reuse_lane(&ucp_ep_config(tmp_ep)->key,
                                                     &ucp_ep_config(ep)->key,
                                                     lane);
            if (found_lane != UCP_NULL_LANE) {
                uct_ep = tmp_ep->uct_eps[lane];
                ucs_assert(ucp_wireup_ep_test(uct_ep) &&
                           !ucp_wireup_ep(uct_ep)->super.is_owner);

                ucs_debug("ep %p: destroy uct_ep[%d]=%p", ep, lane, uct_ep);
                uct_ep_destroy(uct_ep);
                tmp_ep->uct_eps[lane] = NULL;
            }
        }
    }

    if (complete_cb != NULL) {
        param.op_attr_mask |= UCP_OP_ATTR_FIELD_CALLBACK |
                              UCP_OP_ATTR_FIELD_USER_DATA;
        param.cb.send       = complete_cb;
        param.user_data     = wireup_ep;
    }

    wireup_ep->tmp_ep = NULL;

    req = ucp_ep_flush_internal(tmp_ep, ep_flush_flags, 0, &param, NULL,
                                ucp_wireup_tmp_ep_flushed_cb,
                                "tmp_ep_flushed_cb");
    if (req != NULL) {
        if (!UCS_PTR_IS_ERR(req)) {
            ++worker->flush_ops_count;
            return 0;
        }

        ucs_error("ucp_ep_flush_internal() completed with error: %s",
                  ucs_status_string(UCS_PTR_STATUS(req)));
    }

    ucp_ep_disconnected(tmp_ep, 1);
    return 1;
}

void ucp_wireup_ep_replay_pending_requests(ucp_ep_h ucp_ep,
                                           ucs_queue_head_t *tmp_pending_queue)
{
    uct_pending_req_t *uct_req;
    ucp_request_t *req;

    /* Replay pending requests */
    ucs_queue_for_each_extract(uct_req, tmp_pending_queue, priv, 1) {
        req = ucs_container_of(uct_req, ucp_request_t, send.uct);
        ucs_assert(req->send.ep == ucp_ep);
        ucp_request_send(req, 0);
        --ucp_ep->worker->flush_ops_count;
    }
}

/*
 * We switch the endpoint in this function (instead in wireup code) since
 * this is guaranteed to run from the main thread.
 */
static unsigned ucp_wireup_ep_progress(void *arg)
{
    ucp_wireup_ep_t *wireup_ep = arg;
    ucp_ep_h ucp_ep            = wireup_ep->super.ucp_ep;
    ucs_queue_head_t tmp_pending_queue;
    uct_pending_req_t *uct_req;
    int ret;

    UCS_ASYNC_BLOCK(&ucp_ep->worker->async);

    ucs_assert(wireup_ep->flags & UCP_WIREUP_EP_FLAG_READY);
    ucs_assert(wireup_ep->super.uct_ep != NULL);

    /* If we still have pending wireup messages, send them out first */
    if (wireup_ep->pending_count != 0) {
        goto out_unblock;
    }

    /* If an error happened on the endpoint (but perhaps the deferred error handler,
     * ucp_worker_iface_err_handle_progress(), was not called yet, avoid changing
     * ep state, and let the error handler take care of cleanup.
     */
    if (ucp_ep->flags & UCP_EP_FLAG_FAILED) {
        ucs_trace("ep %p: not switching wireup_ep %p to ready state because of error",
                  ucp_ep, wireup_ep);
        goto out_unblock;
    }

    if (wireup_ep->tmp_ep != NULL) {
        ret = ucp_wireup_tmp_ep_destroy(ucp_ep, wireup_ep, UCT_FLUSH_FLAG_LOCAL,
                                        ucp_wireup_tmp_ep_destroy_complete_cb);
        if (!ret) {
            wireup_ep->flags |= UCP_WIREUP_EP_FLAG_DESTROY_TMP_EP;
            goto out_unblock;
        }
    } else if (wireup_ep->flags & UCP_WIREUP_EP_FLAG_DESTROY_TMP_EP) {
        /* destroying of TMP EP is in progress, return from the function */
        goto out_unblock;
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

    UCS_ASYNC_UNBLOCK(&ucp_ep->worker->async);

    /* Replay pending requests */
    ucp_wireup_ep_replay_pending_requests(ucp_ep, &tmp_pending_queue);

    return 0;

out_unblock:
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

    if ((wireup_ep->flags & UCP_WIREUP_EP_FLAG_READY) || (wireup_ep->aux_ep == NULL)) {
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
        ucs_free(proxy_req);
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
        ucs_free((void*)req->send.buffer);
        ucs_free(req);
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
        proxy_req = ucs_malloc(sizeof(*proxy_req), "ucp_wireup_proxy_req");
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
            ucs_free(proxy_req);
        }
    } else {
        ucs_queue_push(&wireup_ep->pending_q, ucp_wireup_ep_req_priv(req));
        ++ucp_ep->worker->flush_ops_count;
        status = UCS_OK;
    }
out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    /* coverity[leaked_storage] */
    return status;
}

static void
ucp_wireup_ep_pending_purge(uct_ep_h uct_ep, uct_pending_purge_callback_t cb,
                            void *arg)
{
    ucp_wireup_ep_t   *wireup_ep = ucp_wireup_ep(uct_ep);
    ucp_worker_h      worker;
    uct_pending_req_t *req;
    ucp_request_t     *ucp_req;

    worker = wireup_ep->super.ucp_ep->worker;

    ucs_queue_for_each_extract(req, &wireup_ep->pending_q, priv, 1) {
        ucp_req = ucs_container_of(req, ucp_request_t, send.uct);
        UCS_ASYNC_BLOCK(&worker->async);
        --worker->flush_ops_count;
        UCS_ASYNC_UNBLOCK(&worker->async);
        cb(&ucp_req->send.uct, arg);
    }

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

ucs_status_t
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

    /* select an auxiliary transport which would be used to pass connection
     * establishment messages.
     */
    status = ucp_wireup_select_aux_transport(ucp_ep, ep_init_flags,
                                             remote_address, &select_info);
    if (status != UCS_OK) {
        return status;
    }

    wireup_ep->aux_rsc_index = select_info.rsc_index;
    aux_addr                 = &remote_address->address_list[select_info.addr_index];
    wiface                   = ucp_worker_iface(worker, select_info.rsc_index);

    /* create auxiliary endpoint connected to the remote iface. */
    uct_ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE    |
                               UCT_EP_PARAM_FIELD_DEV_ADDR |
                               UCT_EP_PARAM_FIELD_IFACE_ADDR;
    uct_ep_params.iface      = wiface->iface;
    uct_ep_params.dev_addr   = aux_addr->dev_addr;
    uct_ep_params.iface_addr = aux_addr->iface_addr;
    status = uct_ep_create(&uct_ep_params, &wireup_ep->aux_ep);
    if (status != UCS_OK) {
        return status;
    }

    ucp_worker_iface_progress_ep(wiface);

    ucs_debug("ep %p: wireup_ep %p created aux_ep %p to %s using "
              UCT_TL_RESOURCE_DESC_FMT, ucp_ep, wireup_ep, wireup_ep->aux_ep,
              ucp_ep_peer_name(ucp_ep),
              UCT_TL_RESOURCE_DESC_ARG(&worker->context->tl_rscs[select_info.rsc_index].tl_rsc));

    return UCS_OK;
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


UCS_CLASS_INIT_FUNC(ucp_wireup_ep_t, ucp_ep_h ucp_ep)
{
    static uct_iface_ops_t ops = {
        .ep_connect_to_ep    = ucp_wireup_ep_connect_to_ep,
        .ep_flush            = ucp_wireup_ep_flush,
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

    self->aux_ep             = NULL;
    self->sockaddr_ep        = NULL;
    self->tmp_ep             = NULL;
    self->aux_rsc_index      = UCP_NULL_RESOURCE;
    self->sockaddr_rsc_index = UCP_NULL_RESOURCE;
    self->pending_count      = 0;
    self->flags              = 0;
    self->progress_id        = UCS_CALLBACKQ_ID_NULL;
    self->cm_idx             = UCP_NULL_RESOURCE;
    ucs_queue_head_init(&self->pending_q);

    UCS_ASYNC_BLOCK(&ucp_ep->worker->async);
    ++ucp_ep->worker->flush_ops_count;
    UCS_ASYNC_UNBLOCK(&ucp_ep->worker->async);

    ucs_trace("ep %p: created wireup ep %p to %s ", ucp_ep, self,
              ucp_ep_peer_name(ucp_ep));
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(ucp_wireup_ep_t)
{
    ucp_ep_h ucp_ep     = self->super.ucp_ep;
    ucp_worker_h worker = ucp_ep->worker;
    int UCS_V_UNUSED ret;

    ucs_assert(ucs_queue_is_empty(&self->pending_q));
    ucs_assert(self->pending_count == 0);

    ucs_debug("ep %p: destroy wireup ep %p", ucp_ep, self);

    uct_worker_progress_unregister_safe(worker->uct, &self->progress_id);
    if (self->aux_ep != NULL) {
        ucp_worker_iface_unprogress_ep(ucp_worker_iface(worker,
                                                        self->aux_rsc_index));
        uct_ep_destroy(self->aux_ep);
    }
    if (self->sockaddr_ep != NULL) {
        uct_ep_destroy(self->sockaddr_ep);
    }

    UCS_ASYNC_BLOCK(&worker->async);
    if (self->tmp_ep != NULL) {
        ucs_assert(!(self->tmp_ep->flags & UCP_EP_FLAG_USED));
        /* TODO: replace by ucp_worker_discard_uct_ep() with FLUSH_CANCEL flag,
         * since the TMP EP will never exist after completing the destroying
         * of the TMP EP */
        ret = ucp_wireup_tmp_ep_destroy(ucp_ep, self, UCT_FLUSH_FLAG_CANCEL, NULL);
        ucs_assert(ret == 1);
    }

    --worker->flush_ops_count;
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

    ucp_proxy_ep_set_uct_ep(&wireup_ep->super, next_ep, 1);

    ucs_debug("ep %p: created next_ep %p to %s using " UCT_TL_RESOURCE_DESC_FMT,
              ucp_ep, wireup_ep->super.uct_ep, ucp_ep_peer_name(ucp_ep),
              UCT_TL_RESOURCE_DESC_ARG(&worker->context->tl_rscs[rsc_index].tl_rsc));

    /* we need to create an auxiliary transport only for active messages */
    if (connect_aux) {
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

static ucs_status_t ucp_wireup_ep_pack_sockaddr_aux_tls(ucp_worker_h worker,
                                                        const char *dev_name,
                                                        uint64_t *tl_bitmap_p,
                                                        ucp_address_t **address_p,
                                                        size_t *address_length_p)
{
    ucp_context_h context = worker->context;
    int tl_id, found_supported_tl = 0;
    ucs_status_t status;
    uint64_t tl_bitmap = 0;

    /* Find a transport which matches the given dev_name and the user's configuration.
     * It also has to be a UCT_IFACE_FLAG_CONNECT_TO_IFACE transport and support
     * active messaging for sending a wireup message */
    ucs_for_each_bit(tl_id, context->config.sockaddr_aux_rscs_bitmap) {
        if ((!strncmp(context->tl_rscs[tl_id].tl_rsc.dev_name, dev_name,
                      UCT_DEVICE_NAME_MAX)) &&
            (ucs_test_all_flags(ucp_worker_iface_get_attr(worker, tl_id)->cap.flags,
                                UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                UCT_IFACE_FLAG_AM_BCOPY))) {
            found_supported_tl = 1;
            tl_bitmap |= UCS_BIT(tl_id);
        }
    }

    if (found_supported_tl) {
        status = ucp_address_pack(worker, NULL, tl_bitmap,
                                  UCP_ADDRESS_PACK_FLAGS_ALL, NULL,
                                  address_length_p, (void**)address_p);
    } else {
        ucs_error("no supported sockaddr auxiliary transports found for %s", dev_name);
        status = UCS_ERR_UNREACHABLE;
    }

    *tl_bitmap_p = tl_bitmap;
    return status;
}

ssize_t ucp_wireup_ep_sockaddr_fill_private_data(void *arg,
                                                 const uct_cm_ep_priv_data_pack_args_t
                                                 *pack_args, void *priv_data)
{
    ucp_wireup_sockaddr_data_t *sa_data = priv_data;
    ucp_wireup_ep_t *wireup_ep          = arg;
    ucp_ep_h ucp_ep                     = wireup_ep->super.ucp_ep;
    ucp_rsc_index_t sockaddr_rsc        = wireup_ep->sockaddr_rsc_index;
    ucp_worker_h worker                 = ucp_ep->worker;
    ucp_context_h context               = worker->context;
    size_t address_length, conn_priv_len;
    ucp_address_t *worker_address, *rsc_address;
    uct_iface_attr_t *attrs;
    ucs_status_t status;
    uint64_t tl_bitmap;
    char aux_tls_str[64];
    const char *dev_name;

    ucs_assert_always(pack_args->field_mask &
                      UCT_CM_EP_PRIV_DATA_PACK_ARGS_FIELD_DEVICE_NAME);

    dev_name = pack_args->dev_name;

    status = ucp_address_pack(worker, NULL, UINT64_MAX,
                              UCP_ADDRESS_PACK_FLAGS_ALL, NULL,
                              &address_length, (void**)&worker_address);
    if (status != UCS_OK) {
        goto err;
    }

    conn_priv_len = sizeof(*sa_data) + address_length;

    /* pack client data */
    ucs_assert((int)ucp_ep_config(ucp_ep)->key.err_mode <= UINT8_MAX);
    sa_data->err_mode  = ucp_ep_config(ucp_ep)->key.err_mode;
    sa_data->ep_id     = ucp_ep_local_id(ucp_ep);
    sa_data->dev_index = UCP_NULL_RESOURCE; /* Not used */

    attrs = ucp_worker_iface_get_attr(worker, sockaddr_rsc);

    /* check private data length limitation */
    if (conn_priv_len > attrs->max_conn_priv) {

        /* since the full worker address is too large to fit into the trasnport's
         * private data, try to pack sockaddr aux tls to pass in the address */
        status = ucp_wireup_ep_pack_sockaddr_aux_tls(worker, dev_name,
                                                     &tl_bitmap, &rsc_address,
                                                     &address_length);
        if (status != UCS_OK) {
            goto err_free_address;
        }

        conn_priv_len = sizeof(*sa_data) + address_length;

        /* check the private data length limitation again, now with partial
         * resources packed (and not the entire worker address) */
        if (conn_priv_len > attrs->max_conn_priv) {
            ucs_error("sockaddr aux resources addresses (%s transports)"
                      " information (%zu) exceeds max_priv on "
                      UCT_TL_RESOURCE_DESC_FMT" (%zu)",
                      ucp_tl_bitmap_str(context, tl_bitmap, aux_tls_str,
                                        sizeof(aux_tls_str)),
                      conn_priv_len,
                      UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[sockaddr_rsc].tl_rsc),
                      attrs->max_conn_priv);
            status = UCS_ERR_UNREACHABLE;
            ucs_free(rsc_address);
            goto err_free_address;
        }

        sa_data->addr_mode = UCP_WIREUP_SA_DATA_PARTIAL_ADDR;
        memcpy(sa_data + 1, rsc_address, address_length);
        ucp_ep->flags |= UCP_EP_FLAG_SOCKADDR_PARTIAL_ADDR;

        ucs_free(rsc_address);

        ucs_trace("sockaddr tl ("UCT_TL_RESOURCE_DESC_FMT") sending partial address: "
                  "(%s transports) (len=%zu) to server. "
                  "total client priv data len: %zu",
                  UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[sockaddr_rsc].tl_rsc),
                  ucp_tl_bitmap_str(context, tl_bitmap, aux_tls_str,
                                    sizeof(aux_tls_str)),
                  address_length, conn_priv_len);
    } else {
        sa_data->addr_mode = UCP_WIREUP_SA_DATA_FULL_ADDR;
        memcpy(sa_data + 1, worker_address, address_length);
    }

    ucp_worker_release_address(worker, worker_address);
    return conn_priv_len;

err_free_address:
    ucp_worker_release_address(worker, worker_address);
err:
    return status;
}

ucs_status_t ucp_wireup_ep_connect_to_sockaddr(uct_ep_h uct_ep,
                                               const ucp_ep_params_t *params)
{
    ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(uct_ep);
    ucp_ep_h ucp_ep            = wireup_ep->super.ucp_ep;
    ucp_worker_h worker        = ucp_ep->worker;
    char saddr_str[UCS_SOCKADDR_STRING_LEN];
    uct_ep_params_t uct_ep_params;
    ucp_rsc_index_t sockaddr_rsc;
    ucp_worker_iface_t *wiface;
    ucs_status_t status;

    ucs_assert(ucp_wireup_ep_test(uct_ep));

    status = ucp_wireup_select_sockaddr_transport(worker->context,
                                                  &params->sockaddr,
                                                  &sockaddr_rsc);
    if (status != UCS_OK) {
        goto out;
    }

    wiface = ucp_worker_iface(worker, sockaddr_rsc);

    wireup_ep->sockaddr_rsc_index = sockaddr_rsc;

    /* Fill parameters and send connection request using the transport */
    uct_ep_params.field_mask        = UCT_EP_PARAM_FIELD_IFACE             |
                                      UCT_EP_PARAM_FIELD_USER_DATA         |
                                      UCT_EP_PARAM_FIELD_SOCKADDR          |
                                      UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS |
                                      UCT_EP_PARAM_FIELD_SOCKADDR_PACK_CB;
    uct_ep_params.iface             = wiface->iface;
    uct_ep_params.sockaddr          = &params->sockaddr;
    uct_ep_params.user_data         = wireup_ep;
    uct_ep_params.sockaddr_cb_flags = UCT_CB_FLAG_ASYNC;
    uct_ep_params.sockaddr_pack_cb  = ucp_wireup_ep_sockaddr_fill_private_data;
    status = uct_ep_create(&uct_ep_params, &wireup_ep->sockaddr_ep);
    if (status != UCS_OK) {
        goto out;
    }

    ucs_debug("ep %p connecting to %s", ucp_ep,
              ucs_sockaddr_str(params->sockaddr.addr, saddr_str, sizeof(saddr_str)));
    status = UCS_OK;

out:
    return status;
}

void ucp_wireup_ep_set_next_ep(uct_ep_h uct_ep, uct_ep_h next_ep)
{
    ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(uct_ep);

    ucs_assert(wireup_ep != NULL);
    ucs_assert(wireup_ep->super.uct_ep == NULL);
    wireup_ep->flags |= UCP_WIREUP_EP_FLAG_LOCAL_CONNECTED;
    ucp_proxy_ep_set_uct_ep(&wireup_ep->super, next_ep, 1);
}

uct_ep_h ucp_wireup_ep_extract_next_ep(uct_ep_h uct_ep)
{
    ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(uct_ep);
    uct_ep_h next_ep;

    ucs_assert_always(wireup_ep != NULL);
    next_ep = wireup_ep->super.uct_ep;
    ucp_proxy_ep_set_uct_ep(&wireup_ep->super, NULL, 0);
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
    ucs_assert(wireup_ep->flags == 0);
}

void ucp_wireup_ep_remote_connected(uct_ep_h uct_ep)
{
    ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(uct_ep);
    ucp_ep_h ucp_ep;

    ucs_assert(wireup_ep != NULL);
    ucs_assert(wireup_ep->super.uct_ep != NULL);
    ucs_assert(wireup_ep->flags & UCP_WIREUP_EP_FLAG_LOCAL_CONNECTED);

    ucp_ep = wireup_ep->super.ucp_ep;

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
    ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(uct_ep);

    if (wireup_ep == NULL) {
        return 0;
    }

    return (wireup_ep->aux_ep == owned_ep) ||
           (wireup_ep->sockaddr_ep == owned_ep) ||
           (wireup_ep->super.uct_ep == owned_ep);
}

void ucp_wireup_ep_disown(uct_ep_h uct_ep, uct_ep_h owned_ep)
{
    ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(uct_ep);

    ucs_assert_always(wireup_ep != NULL);
    if (wireup_ep->aux_ep == owned_ep) {
        wireup_ep->aux_ep = NULL;
    } else if (wireup_ep->sockaddr_ep == owned_ep) {
        wireup_ep->sockaddr_ep = NULL;
    } else if (wireup_ep->super.uct_ep == owned_ep) {
        ucp_proxy_ep_extract(uct_ep);
    }
}

ucp_wireup_ep_t *ucp_wireup_ep(uct_ep_h uct_ep)
{
    return ucp_wireup_ep_test(uct_ep) ?
           ucs_derived_of(uct_ep, ucp_wireup_ep_t) : NULL;
}
