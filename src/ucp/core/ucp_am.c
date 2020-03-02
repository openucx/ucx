/**
* Copyright (C) Los Alamos National Security, LLC. 2019 ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
* 
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucp_am.h"
#include "ucp_am.inl"

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_context.h>
#include <ucp/proto/proto_am.inl>
#include <ucp/dt/dt.h>
#include <ucp/dt/dt.inl>


void ucp_am_ep_init(ucp_ep_h ep)
{
    ucp_ep_ext_proto_t *ep_ext = ucp_ep_ext_proto(ep);
    
    if (ep->worker->context->config.features & UCP_FEATURE_AM) {
        ucs_list_head_init(&ep_ext->am.started_ams);
    }
}

void ucp_am_ep_cleanup(ucp_ep_h ep)
{
    ucp_ep_ext_proto_t *ep_ext = ucp_ep_ext_proto(ep);

    if (ep->worker->context->config.features & UCP_FEATURE_AM) {
        if (ucs_unlikely(!ucs_list_is_empty(&ep_ext->am.started_ams))) {
            ucs_warn("worker : %p not all UCP active messages have been" 
                     "run to completion", ep->worker);
        }
    }
}

UCS_PROFILE_FUNC_VOID(ucp_am_data_release,
                      (worker, data),
                      ucp_worker_h worker, void *data)
{
    ucp_recv_desc_t *rdesc = (ucp_recv_desc_t *)data - 1;
    ucp_recv_desc_t *desc;

    if (rdesc->flags & UCP_RECV_DESC_FLAG_MALLOC) {
        ucs_free(rdesc);
        return;
    } else if (rdesc->flags & UCP_RECV_DESC_FLAG_AM_HDR) {
        desc = rdesc;
        rdesc = UCS_PTR_BYTE_OFFSET(rdesc, -sizeof(ucp_am_hdr_t));
        *rdesc = *desc;
    } else if (rdesc->flags & UCP_RECV_DESC_FLAG_AM_REPLY) {
        desc = rdesc;
        rdesc = UCS_PTR_BYTE_OFFSET(rdesc, -sizeof(ucp_am_reply_hdr_t));
        *rdesc = *desc;
    } 
    ucp_recv_desc_release(rdesc);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_worker_set_am_handler,
                 (worker, id, cb, arg, flags),
                 ucp_worker_h worker, uint16_t id, 
                 ucp_am_callback_t cb, void *arg, 
                 uint32_t flags)
{
    size_t num_entries;

    UCP_CONTEXT_CHECK_FEATURE_FLAGS(worker->context, UCP_FEATURE_AM,
                                    return UCS_ERR_INVALID_PARAM);

    if (id >= worker->am_cb_array_len) {
        num_entries = ucs_align_up_pow2(id + 1, UCP_AM_CB_BLOCK_SIZE);
        worker->am_cbs = ucs_realloc(worker->am_cbs, num_entries * 
                                     sizeof(ucp_worker_am_entry_t),
                                     "UCP AM callback array");
        memset(worker->am_cbs + worker->am_cb_array_len, 
               0, (num_entries - worker->am_cb_array_len)
               * sizeof(ucp_worker_am_entry_t));
        
        worker->am_cb_array_len = num_entries;
    }

    worker->am_cbs[id].cb      = cb;
    worker->am_cbs[id].context = arg;
    worker->am_cbs[id].flags   = flags;

    return UCS_OK;
}

static size_t 
ucp_am_bcopy_pack_args_single(void *dest, void *arg)
{
    ucp_am_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    ucs_assert(req->send.state.dt.offset == 0);
    
    hdr->am_hdr.am_id  = req->send.am.am_id;
    hdr->am_hdr.length = req->send.length;
    hdr->am_hdr.flags  = req->send.am.flags;

    length = ucp_dt_pack(req->send.ep->worker, req->send.datatype,
                         UCS_MEMORY_TYPE_HOST, hdr + 1, req->send.buffer,
                         &req->send.state.dt, req->send.length);
    ucs_assert(length == req->send.length);

    return sizeof(*hdr) + length;
}

static size_t 
ucp_am_bcopy_pack_args_single_reply(void *dest, void *arg)
{
    ucp_am_reply_hdr_t *reply_hdr = dest;
    ucp_request_t *req = arg;
    size_t length;
    size_t hdr_size;

    ucs_assert(req->send.state.dt.offset == 0);
    
    reply_hdr->super.am_hdr.am_id  = req->send.am.am_id;
    reply_hdr->super.am_hdr.length = req->send.length;
    reply_hdr->super.am_hdr.flags  = req->send.am.flags;
    reply_hdr->ep_ptr              = ucp_request_get_dest_ep_ptr(req);

    length = ucp_dt_pack(req->send.ep->worker, req->send.datatype,
                         UCS_MEMORY_TYPE_HOST, reply_hdr + 1,
                         req->send.buffer,
                         &req->send.state.dt, req->send.length);
    hdr_size = sizeof(*reply_hdr);
    ucs_assert(length == req->send.length);

    return hdr_size + length;
}

static size_t 
ucp_am_bcopy_pack_args_first(void *dest, void *arg)
{
    ucp_am_long_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;
    
    length = ucp_ep_get_max_bcopy(req->send.ep, req->send.lane) -
                                  sizeof(*hdr);
    hdr->total_size = req->send.length;
    hdr->am_id      = req->send.am.am_id;
    hdr->msg_id     = req->send.am.message_id;
    hdr->ep         = ucp_request_get_dest_ep_ptr(req);
    hdr->offset     = req->send.state.dt.offset;
    
    ucs_assert(req->send.state.dt.offset == 0);
    ucs_assert(req->send.length > length);

    return sizeof(*hdr) + ucp_dt_pack(req->send.ep->worker, 
                                      req->send.datatype, 
                                      UCS_MEMORY_TYPE_HOST,
                                      hdr + 1, req->send.buffer,
                                      &req->send.state.dt, length);
}

static size_t 
ucp_am_bcopy_pack_args_mid(void *dest, void *arg)
{
    ucp_am_long_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;
    size_t max_bcopy;

    max_bcopy = ucp_ep_get_max_bcopy(req->send.ep, req->send.lane);
    length    = ucs_min(max_bcopy - sizeof(*hdr),
                        req->send.length - req->send.state.dt.offset);

    hdr->msg_id     = req->send.am.message_id;
    hdr->offset     = req->send.state.dt.offset;
    hdr->ep         = ucp_request_get_dest_ep_ptr(req);
    hdr->am_id      = req->send.am.am_id;
    hdr->total_size = req->send.length;
    
    return sizeof(*hdr) + ucp_dt_pack(req->send.ep->worker,
                                      req->send.datatype,
                                      UCS_MEMORY_TYPE_HOST,
                                      hdr + 1, req->send.buffer,
                                      &req->send.state.dt, length);
}

static ucs_status_t ucp_am_send_short(ucp_ep_h ep, uint16_t id, 
                                      const void *payload, size_t length)
{
    uct_ep_h am_ep = ucp_ep_get_am_uct_ep(ep);
    ucp_am_hdr_t hdr;

    hdr.am_hdr.am_id  = id;
    hdr.am_hdr.length = length;
    hdr.am_hdr.flags  = 0;
    ucs_assert(sizeof(ucp_am_hdr_t) == sizeof(uint64_t));
    
    return uct_ep_am_short(am_ep, UCP_AM_ID_SINGLE, hdr.u64, 
                           (void *)payload, length);
}

static ucs_status_t ucp_am_contig_short(uct_pending_req_t *self)
{
    ucp_request_t *req   = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_status_t  status = ucp_am_send_short(req->send.ep, 
                                             req->send.am.am_id, 
                                             req->send.buffer, 
                                             req->send.length);
    if (ucs_likely(status == UCS_OK)) {
        ucp_request_complete_send(req, UCS_OK);
    }

    return status;
}

static ucs_status_t ucp_am_bcopy_single(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_status_t status;
    
    status = ucp_do_am_bcopy_single(self, UCP_AM_ID_SINGLE, 
                                    ucp_am_bcopy_pack_args_single);
    if (status == UCS_OK) {
        ucp_request_send_generic_dt_finish(req);
        ucp_request_complete_send(req, UCS_OK);
    }

    return status;
}

static ucs_status_t ucp_am_bcopy_single_reply(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_status_t status;
    
    status = ucp_do_am_bcopy_single(self, UCP_AM_ID_SINGLE_REPLY,
                                    ucp_am_bcopy_pack_args_single_reply);
    if (status == UCS_OK) {
        ucp_request_send_generic_dt_finish(req);
        ucp_request_complete_send(req, UCS_OK);
    }

    return status;
}

static ucs_status_t ucp_am_bcopy_multi(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_multi(self, UCP_AM_ID_MULTI,
                                                UCP_AM_ID_MULTI,
                                                ucp_am_bcopy_pack_args_first,
                                                ucp_am_bcopy_pack_args_mid, 0);
    ucp_request_t *req;
    
    if (status == UCS_OK) {
        req = ucs_container_of(self, ucp_request_t, send.uct);
        ucp_request_send_generic_dt_finish(req);
        ucp_request_complete_send(req, UCS_OK);
    } else if (status == UCP_STATUS_PENDING_SWITCH) {
        status = UCS_OK;
    }

    return status;
}

static ucs_status_t ucp_am_bcopy_multi_reply(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_multi(self, UCP_AM_ID_MULTI_REPLY,
                                                UCP_AM_ID_MULTI_REPLY,
                                                ucp_am_bcopy_pack_args_first,
                                                ucp_am_bcopy_pack_args_mid, 0);
    ucp_request_t *req;

    if (status == UCS_OK) {
        req = ucs_container_of(self, ucp_request_t, send.uct);
        ucp_request_send_generic_dt_finish(req);
        ucp_request_complete_send(req, UCS_OK);
    } else if (status == UCP_STATUS_PENDING_SWITCH) {
        status = UCS_OK;
    }

    return status;
}

static ucs_status_t ucp_am_zcopy_single(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_am_hdr_t hdr;
    
    hdr.am_hdr.am_id  = req->send.am.am_id;
    hdr.am_hdr.length = req->send.length;
    hdr.am_hdr.flags  = req->send.am.flags;
    
    return ucp_do_am_zcopy_single(self, UCP_AM_ID_SINGLE, &hdr,
                                  sizeof(hdr), ucp_proto_am_zcopy_req_complete);
}

static ucs_status_t ucp_am_zcopy_single_reply(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_am_reply_hdr_t reply_hdr;

    reply_hdr.super.am_hdr.am_id  = req->send.am.am_id;
    reply_hdr.super.am_hdr.length = req->send.length;
    reply_hdr.super.am_hdr.flags  = req->send.am.flags;
    reply_hdr.ep_ptr              = ucp_request_get_dest_ep_ptr(req);
    
    return ucp_do_am_zcopy_single(self, UCP_AM_ID_SINGLE_REPLY, 
                                  &reply_hdr, sizeof(reply_hdr), 
                                  ucp_proto_am_zcopy_req_complete);
}

static ucs_status_t ucp_am_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_am_long_hdr_t hdr;
    
    hdr.ep         = ucp_request_get_dest_ep_ptr(req);
    hdr.msg_id     = req->send.am.message_id;
    hdr.offset     = req->send.state.dt.offset;
    hdr.am_id      = req->send.am.am_id;
    hdr.total_size = req->send.length;
    
    return ucp_do_am_zcopy_multi(self, UCP_AM_ID_MULTI, 
                                 UCP_AM_ID_MULTI,
                                 &hdr, sizeof(hdr),
                                 &hdr, sizeof(hdr),
                                 ucp_proto_am_zcopy_req_complete, 0);
}

static ucs_status_t ucp_am_zcopy_multi_reply(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_am_long_hdr_t hdr;

    hdr.ep         = ucp_request_get_dest_ep_ptr(req);
    hdr.msg_id     = req->send.am.message_id;
    hdr.offset     = req->send.state.dt.offset;
    hdr.am_id      = req->send.am.am_id;
    hdr.total_size = req->send.length;
    
    return ucp_do_am_zcopy_multi(self, UCP_AM_ID_MULTI_REPLY, 
                                 UCP_AM_ID_MULTI_REPLY,
                                 &hdr, sizeof(hdr),
                                 &hdr, sizeof(hdr),
                                 ucp_proto_am_zcopy_req_complete, 0);
}

static void ucp_am_send_req_init(ucp_request_t *req, ucp_ep_h ep,
                                 const void *buffer, uintptr_t datatype,
                                 size_t count, uint16_t flags, 
                                 uint16_t am_id)
{
    req->flags          = UCP_REQUEST_FLAG_SEND_AM;
    req->send.ep        = ep;
    req->send.am.am_id  = am_id;
    req->send.am.flags  = flags;
    req->send.buffer    = (void *) buffer;
    req->send.datatype  = datatype;
    req->send.mem_type  = UCS_MEMORY_TYPE_HOST;
    req->send.lane      = ep->am_lane;

    ucp_request_send_state_init(req, datatype, count);
    req->send.length = ucp_dt_length(req->send.datatype, count,
                                     req->send.buffer,
                                     &req->send.state.dt);
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_am_send_req(ucp_request_t *req, size_t count,
                const ucp_ep_msg_config_t *msg_config,
                ucp_send_callback_t cb, const ucp_request_send_proto_t *proto)
{
    
    size_t zcopy_thresh = ucp_proto_get_zcopy_threshold(req, msg_config,
                                                        count, SIZE_MAX);
    ssize_t max_short   = ucp_am_get_short_max(req, msg_config);
    ucs_status_t status;
    
    status = ucp_request_send_start(req, max_short, 
                                    zcopy_thresh, SIZE_MAX,
                                    count, msg_config,
                                    proto);
    if (status != UCS_OK) {
       return UCS_STATUS_PTR(status);
    }

    /* Start the request.
     * If it is completed immediately, release the request and return the status.
     * Otherwise, return the request.
     */
    status = ucp_request_send(req, 0);
    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        ucs_trace_req("releasing send request %p, returning status %s", req,
                      ucs_status_string(status));
        ucp_request_put(req);
        return UCS_STATUS_PTR(status);
    }

    ucp_request_set_callback(req, send.cb, cb);
    
    return req + 1;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_am_send_nb, 
                 (ep, id, payload, count, datatype, cb, flags),
                 ucp_ep_h ep, uint16_t id, const void *payload, 
                 size_t count, uintptr_t datatype, 
                 ucp_send_callback_t cb, unsigned flags)
{
    ucs_status_t status;
    ucs_status_ptr_t ret;
    ucp_request_t *req;
    size_t length;
    
    UCP_CONTEXT_CHECK_FEATURE_FLAGS(ep->worker->context, UCP_FEATURE_AM,
                                    return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM));

    if (ucs_unlikely((flags != 0) && !(flags & UCP_AM_SEND_REPLY))) {
        return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM);
    }

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    if (ucs_likely(!(flags & UCP_AM_SEND_REPLY)) && 
        (ucs_likely(UCP_DT_IS_CONTIG(datatype)))) {
        length = ucp_contig_dt_length(datatype, count);
        
        if (ucs_likely((ssize_t)length <= ucp_ep_config(ep)->am.max_short)) {
            status = ucp_am_send_short(ep, id, payload, length);
            if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
                UCP_EP_STAT_TAG_OP(ep, EAGER);
                ret = UCS_STATUS_PTR(status);
                goto out;
            }
        }
    }

    req = ucp_request_get(ep->worker);
    if (ucs_unlikely(req == NULL)) {
        ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
        goto out;
    }

    ucp_am_send_req_init(req, ep, payload, datatype, count, flags, id);
    status = ucp_ep_resolve_dest_ep_ptr(ep, ep->am_lane);
    if (ucs_unlikely(status != UCS_OK)) {
        ret = UCS_STATUS_PTR(status);
        goto out;
    }

    if (flags & UCP_AM_SEND_REPLY) {
        ret = ucp_am_send_req(req, count, &ucp_ep_config(ep)->am, cb,
                              ucp_ep_config(ep)->am_u.reply_proto);
    } else {
        ret = ucp_am_send_req(req, count, &ucp_ep_config(ep)->am, cb,
                              ucp_ep_config(ep)->am_u.proto);
    }

out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    return ret;
}

static ucs_status_t
ucp_am_handler_common(ucp_worker_h worker, void *hdr_end,
                      size_t hdr_size, size_t args_length,
                      ucp_ep_h reply_ep, uint16_t am_id, 
                      uint16_t desc_flag, unsigned am_flags) 
{
    ucp_recv_desc_t *desc = NULL;
    uint16_t recv_flags = 0;
    ucs_status_t status;

    if (ucs_unlikely((am_id >= worker->am_cb_array_len) ||
                     (worker->am_cbs[am_id].cb == NULL))) {
        ucs_warn("UCP Active Message was received with id : %u, but there" 
                 "is no registered callback for that id", am_id);
        return UCS_OK;
    }

    if (ucs_unlikely(am_flags & UCT_CB_PARAM_FLAG_DESC)) {
        recv_flags |= desc_flag;
    }

    /* TODO find way to do this without rewriting header if 
     * UCT_CB_PARAM_FLAG_DESC flag is set
     */
    status = ucp_recv_desc_init(worker, hdr_end, hdr_size + args_length,
                                0, am_flags, 0,
                                recv_flags, 0, &desc);
    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        ucs_error("worker %p  could not allocate descriptor for active message"
                  "on callback : %u", worker, am_id);
        return status;
    }

    ucs_assert(desc != NULL);

    status = worker->am_cbs[am_id].cb(worker->am_cbs[am_id].context,
                                      desc + 1,
                                      args_length,
                                      reply_ep,
                                      UCP_CB_PARAM_FLAG_DATA);
    if (ucs_unlikely(am_flags & UCT_CB_PARAM_FLAG_DESC)) {
        return status;
    }
    
    if (status == UCS_OK) {
        ucp_recv_desc_release(desc);
        return UCS_OK;
    } else if (status == UCS_INPROGRESS) {
        return UCS_OK;
    } 

    return status;
}

static ucs_status_t 
ucp_am_handler_reply(void *am_arg, void *am_data, size_t am_length,
                     unsigned am_flags)
{
    ucp_am_reply_hdr_t *hdr = (ucp_am_reply_hdr_t *)am_data;
    ucp_worker_h worker     = (ucp_worker_h)am_arg;
    uint16_t am_id          = hdr->super.am_hdr.am_id;
    ucp_ep_h reply_ep;

    reply_ep = ucp_worker_get_ep_by_ptr(worker, hdr->ep_ptr);
 
    return ucp_am_handler_common(worker, hdr + 1, sizeof(*hdr),
                                 am_length - sizeof(*hdr), reply_ep,
                                 am_id, UCP_RECV_DESC_FLAG_AM_REPLY, 
                                 am_flags);
}

static ucs_status_t 
ucp_am_handler(void *am_arg, void *am_data, size_t am_length,
               unsigned am_flags)
{
    ucp_worker_h worker   = (ucp_worker_h)am_arg;
    ucp_am_hdr_t *hdr     = (ucp_am_hdr_t *)am_data;
    uint16_t am_id        = hdr->am_hdr.am_id;

    return ucp_am_handler_common(worker, hdr + 1, sizeof(*hdr),
                                 am_length - sizeof(*hdr), NULL,
                                 am_id, UCP_RECV_DESC_FLAG_AM_HDR, 
                                 am_flags);    
}

static ucp_am_unfinished_t *
ucp_am_find_unfinished(ucp_worker_h worker, ucp_ep_h ep, 
                       ucp_ep_ext_proto_t *ep_ext, 
                       ucp_am_long_hdr_t *hdr, size_t am_length)
{
    ucp_am_unfinished_t *unfinished;
    /* TODO make this hash table for faster lookup */
    ucs_list_for_each(unfinished, &ep_ext->am.started_ams, list) {
        if (unfinished->msg_id == hdr->msg_id) {
            return unfinished;
        }
    }
    
    return NULL;
}

static ucs_status_t
ucp_am_handle_unfinished(ucp_worker_h worker,
                         ucp_am_unfinished_t *unfinished, 
                         ucp_am_long_hdr_t *long_hdr, 
                         size_t am_length, ucp_ep_h reply_ep) 
{
    uint16_t am_id;
    ucs_status_t status;

    memcpy(UCS_PTR_BYTE_OFFSET(unfinished->all_data + 1, long_hdr->offset),
           long_hdr + 1, am_length - sizeof(*long_hdr));
    unfinished->left -= am_length - sizeof(*long_hdr);
    if (unfinished->left == 0) {
        am_id = long_hdr->am_id;
        status = worker->am_cbs[am_id].cb(worker->am_cbs[am_id].context,
                                          unfinished->all_data + 1,
                                          long_hdr->total_size,
                                          reply_ep,
                                          UCP_CB_PARAM_FLAG_DATA);

        if (status != UCS_INPROGRESS) {
            ucs_free(unfinished->all_data);
        }

        ucs_list_del(&unfinished->list);
        ucs_free(unfinished);
    }
    
    return UCS_OK;
}

static ucs_status_t
ucp_am_long_handler_common(void *am_arg, void *am_data, size_t am_length,
                           unsigned am_flags, ucp_ep_h reply_ep)
{
    ucp_worker_h worker         = (ucp_worker_h)am_arg;
    ucp_am_long_hdr_t *long_hdr = (ucp_am_long_hdr_t *)am_data;
    ucp_ep_h ep                 = ucp_worker_get_ep_by_ptr(worker, 
                                                           long_hdr->ep);
    ucp_ep_ext_proto_t *ep_ext  = ucp_ep_ext_proto(ep);
    ucp_recv_desc_t *all_data;
    size_t left;
    ucp_am_unfinished_t *unfinished;

    if (ucs_unlikely((long_hdr->am_id >= worker->am_cb_array_len) ||
                     (worker->am_cbs[long_hdr->am_id].cb == NULL))) {
        ucs_warn("UCP Active Message was received with id : %u, but there" 
                 "is no registered callback for that id", long_hdr->am_id);
        return UCS_OK;
    }

    /* if there are multiple messages,
     * we first check to see if any of the other messages
     * have arrived. If any messages have arrived,
     * we copy ourselves into the buffer and leave
     */
    unfinished = ucp_am_find_unfinished(worker, ep, ep_ext, long_hdr, am_length);
    
    if (unfinished) {
        return ucp_am_handle_unfinished(worker, unfinished, 
                                        long_hdr, am_length,
                                        reply_ep);
    }
    
    /* If I am first, I make the buffer for everyone to go into,
     * copy myself in, and put myself on the list so people can find me
     */
    all_data = ucs_malloc(long_hdr->total_size + sizeof(ucp_recv_desc_t),
                          "ucp recv desc for long AM");
    if (ucs_unlikely(all_data == NULL)) {
        return UCS_ERR_NO_MEMORY;
    }

    all_data->flags = UCP_RECV_DESC_FLAG_MALLOC;

    left = long_hdr->total_size - (am_length -
                                   sizeof(ucp_am_long_hdr_t));
    
    memcpy(UCS_PTR_BYTE_OFFSET(all_data + 1, long_hdr->offset),
           long_hdr + 1, am_length - sizeof(ucp_am_long_hdr_t));
    
    /* Can't use a desc for this because of the buffer */
    unfinished           = ucs_malloc(sizeof(ucp_am_unfinished_t),
                                         "unfinished UCP AM");
    if (ucs_unlikely(unfinished == NULL)) {
        ucs_free(all_data);
        return UCS_ERR_NO_MEMORY;
    }

    unfinished->all_data = all_data;
    unfinished->left     = left;
    unfinished->msg_id   = long_hdr->msg_id;

    ucs_list_add_head(&ep_ext->am.started_ams, &unfinished->list);

    return UCS_OK;
}

static ucs_status_t
ucp_am_long_handler_reply(void *am_arg, void *am_data, size_t am_length,
                          unsigned am_flags)
{ 
    ucp_worker_h worker         = (ucp_worker_h)am_arg;
    ucp_am_long_hdr_t *long_hdr = (ucp_am_long_hdr_t *)am_data;
    ucp_ep_h ep                 = ucp_worker_get_ep_by_ptr(worker, 
                                                           long_hdr->ep);
    
    return ucp_am_long_handler_common(am_arg, am_data, am_length, am_flags,
                                      ep);
}

static ucs_status_t
ucp_am_long_handler(void *am_arg, void *am_data, size_t am_length,
                    unsigned am_flags)
{
    return ucp_am_long_handler_common(am_arg, am_data, am_length, am_flags,
                                      NULL); 
}

UCP_DEFINE_AM(UCP_FEATURE_AM, UCP_AM_ID_SINGLE,
              ucp_am_handler, NULL, 0);
UCP_DEFINE_AM(UCP_FEATURE_AM, UCP_AM_ID_MULTI,
              ucp_am_long_handler, NULL, 0);
UCP_DEFINE_AM(UCP_FEATURE_AM, UCP_AM_ID_SINGLE_REPLY,
              ucp_am_handler_reply, NULL, 0);
UCP_DEFINE_AM(UCP_FEATURE_AM, UCP_AM_ID_MULTI_REPLY,
              ucp_am_long_handler_reply, NULL, 0);

const ucp_request_send_proto_t ucp_am_proto = {
    .contig_short           = ucp_am_contig_short,
    .bcopy_single           = ucp_am_bcopy_single,
    .bcopy_multi            = ucp_am_bcopy_multi,
    .zcopy_single           = ucp_am_zcopy_single,
    .zcopy_multi            = ucp_am_zcopy_multi,
    .zcopy_completion       = ucp_proto_am_zcopy_completion,
    .only_hdr_size          = sizeof(ucp_am_hdr_t)
};

const ucp_request_send_proto_t ucp_am_reply_proto = {
    .contig_short           = NULL,
    .bcopy_single           = ucp_am_bcopy_single_reply,
    .bcopy_multi            = ucp_am_bcopy_multi_reply,
    .zcopy_single           = ucp_am_zcopy_single_reply,
    .zcopy_multi            = ucp_am_zcopy_multi_reply,
    .zcopy_completion       = ucp_proto_am_zcopy_completion,
    .only_hdr_size          = sizeof(ucp_am_reply_hdr_t)
};
