/**
* Copyright (C) Los Alamos National Security, LLC. 2018 ALL RIGHTS RESERVED.
* 
* See file LICENSE for terms.
*/

#include "ucp_am.h"

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_context.h>
#include <ucp/proto/proto.h>
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

    if(ep->worker->context->config.features & UCP_FEATURE_AM) {
        ucs_list_del(&ep_ext->am.started_ams);
    }
}

void ucp_am_data_release(ucp_worker_h worker, void *data)
{
    ucp_recv_desc_t *rdesc = (ucp_recv_desc_t *) data - 1;

    if (rdesc->flags & UCP_RECV_DESC_FLAG_MALLOC) {
        ucs_free(rdesc);
        return;
    } else if (rdesc->flags & UCP_RECV_DESC_FLAG_HDR) {
        ucp_recv_desc_t *desc = rdesc;
        rdesc = UCS_PTR_BYTE_OFFSET(rdesc, -sizeof(ucp_am_hdr_t));
        *rdesc = *desc;
    }
    ucp_recv_desc_release(rdesc);
}

ucs_status_t ucp_worker_set_am_handler(ucp_worker_h worker, uint16_t id, 
                                       ucp_am_callback_t cb, void *arg, 
                                       uint32_t flags)
{
    if (id >= worker->am_cb_array_len) {
        size_t num_entries = id + (id % AM_BLOCK);
        worker->am_cbs = ucs_realloc(worker->am_cbs, num_entries * 
                                     sizeof(ucp_worker_am_entry_t),
                                     "UCP AM callback array");
        
        memset(worker->am_cbs + worker->am_cb_array_len, 
               0, (num_entries - worker->am_cb_array_len)
               * sizeof(ucp_worker_am_entry_t));
        
        worker->am_cb_array_len = num_entries;
    }
    worker->am_cbs[id].cb = cb;
    worker->am_cbs[id].context = arg;
    worker->am_cbs[id].flags = flags;

    return UCS_OK;
}

size_t bcopy_pack_args_single(void *dest, void *arg)
{
    ucp_am_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    ucs_assert(req->send.state.dt.offset == 0);

    hdr->am_id = req->send.am.am_id;
    hdr->total_size = req->send.length;

    length = ucp_dt_pack(req->send.ep->worker, req->send.datatype,
                         UCT_MD_MEM_TYPE_HOST, hdr + 1, req->send.buffer,
                         &req->send.state.dt, req->send.length);
    ucs_assert(length == req->send.length);

    return sizeof(ucp_am_hdr_t) + length;
}

size_t bcopy_pack_args_first(void *dest, void *arg){
    ucp_am_long_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;

    length = ucp_ep_get_max_bcopy(req->send.ep, req->send.lane) -
                                  sizeof(ucp_am_long_hdr_t);
    hdr->total_size = req->send.length;
    hdr->am_id      = req->send.am.am_id;
    hdr->msg_id         = req->send.tag.message_id;
    hdr->ep             = ucp_request_get_dest_ep_ptr(req);
    hdr->offset         = req->send.state.dt.offset;
    return sizeof(*hdr) + ucp_dt_pack(req->send.ep->worker, 
                                      req->send.datatype, 
                                      UCT_MD_MEM_TYPE_HOST,
                                      hdr + 1, req->send.buffer,
                                      &req->send.state.dt, length);
                                                    
}

size_t bcopy_pack_args_mid(void *dest, void *arg)
{
    ucp_am_long_hdr_t *hdr = dest;
    ucp_request_t *req = arg;
    size_t length;
    length = ucs_min(ucp_ep_get_max_bcopy(req->send.ep, req->send.lane) - 
                     sizeof(*hdr),
                     req->send.length - req->send.state.dt.offset);

    hdr->msg_id         = req->send.tag.message_id;
    hdr->offset         = req->send.state.dt.offset;
    hdr->ep             = ucp_request_get_dest_ep_ptr(req);
    hdr->am_id      = req->send.am.am_id;
    hdr->total_size = req->send.length;
    return sizeof(*hdr) + ucp_dt_pack(req->send.ep->worker,
                                      req->send.datatype,
                                      UCT_MD_MEM_TYPE_HOST,
                                      hdr + 1, req->send.buffer,
                                      &req->send.state.dt, length);
}

ucs_status_t ucp_am_send_short(ucp_ep_h ep, uint16_t id, 
                               const void *payload, size_t length)
{
    uct_ep_h am_ep = ucp_ep_get_am_uct_ep(ep);
    ucp_am_hdr_t hdr;
    uint64_t *header;
    void *buf;

    hdr.am_id = id;
    hdr.total_size = length;

    ucs_assert(sizeof(ucp_am_hdr_t) == sizeof(uint64_t));

    buf = (void *) payload;
    header = (uint64_t *) &hdr;
    return uct_ep_am_short(am_ep, UCP_AM_ID_SINGLE, *header, buf, length);
}

static ucs_status_t ucp_am_contig_short(uct_pending_req_t *self)
{
    ucp_request_t *req   = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_status_t  status = ucp_am_send_short(req->send.ep, 
                                             req->send.am.am_id, 
                                             req->send.buffer, 
                                             req->send.length);
  
    if (ucs_likely(status == UCS_OK)){
        ucp_request_complete_send(req, UCS_OK);
    }
    return status;
}

static ucs_status_t ucp_am_bcopy_single(uct_pending_req_t *self)
{
    ucs_status_t status;
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    status = ucp_do_am_bcopy_single(self, UCP_AM_ID_SINGLE, 
                                    bcopy_pack_args_single);
    if (status == UCS_OK){
        ucp_request_send_generic_dt_finish(req);
        ucp_request_complete_send(req, UCS_OK);
    }

    return status;
}

static ucs_status_t ucp_am_bcopy_multi(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_multi(self, UCP_AM_ID_MULTI,
                                                UCP_AM_ID_MULTI, 
                                                sizeof(ucp_am_long_hdr_t),
                                                bcopy_pack_args_first,
                                                bcopy_pack_args_mid, 0);
    
    if (status == UCS_OK) {
        ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
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
    ucp_am_hdr_t   hdr;

    hdr.am_id = req->send.am.am_id;
    hdr.total_size = req->send.length;
    
    return ucp_do_am_zcopy_single(self, UCP_AM_ID_SINGLE, &hdr,
                                  sizeof(hdr), ucp_proto_am_zcopy_req_complete);
}

static ucs_status_t ucp_am_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_am_long_hdr_t hdr;

    hdr.ep             = ucp_request_get_dest_ep_ptr(req);
    hdr.msg_id         = req->send.tag.message_id;
    hdr.offset         = req->send.state.dt.offset;
    hdr.am_id      = req->send.am.am_id;
    hdr.total_size = req->send.length;
    return ucp_do_am_zcopy_multi(self, UCP_AM_ID_MULTI, UCP_AM_ID_MULTI,
                                 &hdr, sizeof(hdr),
                                 &hdr, sizeof(hdr),
                                 ucp_proto_am_zcopy_req_complete, 0);
}

static void ucp_am_send_req_init(ucp_request_t *req, ucp_ep_h ep,
                                 const void *buffer, uintptr_t datatype,
                                 size_t count, uint16_t flags, 
                                 uint16_t am_id)
{
    req->flags = flags;
    req->send.ep = ep;
    req->send.am.am_id = am_id;
    req->send.buffer = buffer;
    req->send.datatype = datatype;
    req->send.mem_type = UCT_MD_MEM_TYPE_HOST;
    req->send.lane = ep->am_lane;
    ucp_request_send_state_init(req, datatype, count);
    req->send.length = ucp_dt_length(req->send.datatype, count,
                                     req->send.buffer,
                                     &req->send.state.dt);
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_am_send_req(ucp_request_t *req, size_t count,
                const ucp_ep_msg_config_t * msg_config,
                ucp_send_callback_t cb, const ucp_proto_t *proto)
{
    size_t zcopy_thresh = ucp_proto_get_zcopy_threshold(req, msg_config,
                                                        count, SIZE_MAX);
    ssize_t max_short   = ucp_proto_get_short_max(req, msg_config);
    
    ucs_status_t status = ucp_request_send_start(req, max_short, 
                                                 zcopy_thresh, SIZE_MAX,
                                                 count, msg_config,
                                                 proto);
    if(status != UCS_OK){
       return UCS_STATUS_PTR(status);
    }

    /*Start the request.
     * If it is completed immediately, release the request and return the status.
     * Otherwise, return the request.
     */
    status = ucp_request_send(req);
    if(req->flags & UCP_REQUEST_FLAG_COMPLETED){
        ucs_trace_req("releasing send request %p, returning status %s", req,
                      ucs_status_string(status));
        ucp_request_put(req);
        return UCS_STATUS_PTR(status);
    }

    ucp_request_set_callback(req, send.cb, cb);
    ucs_trace_req("returning send request %p", req);
    
    return req + 1;
}

ucs_status_ptr_t ucp_am_send_nb(ucp_ep_h ep, uint16_t id,
                                const void *payload, size_t count,
                                uintptr_t datatype, ucp_send_callback_t cb,
                                unsigned flags)
{
    ucs_status_t status;
    ucs_status_ptr_t ret;
    size_t length;
    ucp_request_t *req;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);
    
    if(ucs_unlikely(flags != 0)){
        ret = UCS_STATUS_PTR(UCS_ERR_NOT_IMPLEMENTED);
        goto out;
    }

    status = ucp_ep_resolve_dest_ep_ptr(ep, ep->am_lane);
    if (status != UCS_OK) {
        ret = UCS_STATUS_PTR(status);
        goto out;
    }
    if(ucs_likely(UCP_DT_IS_CONTIG(datatype))){
        length = ucp_contig_dt_length(datatype, count);
        if(ucs_likely((ssize_t)length <= ucp_ep_config(ep)->am.max_short)){
            status = ucp_am_send_short(ep, id, payload, length);
            if(ucs_likely(status != UCS_ERR_NO_RESOURCE)){
                UCP_EP_STAT_TAG_OP(ep, EAGER);
                ret = UCS_STATUS_PTR(status);
                goto out;
            }
        }
    }
    req = ucp_request_get(ep->worker);
    if(ucs_unlikely(req == NULL)){
        ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
        goto out;
    }

    ucp_am_send_req_init(req, ep, payload, datatype, count, flags, id);
  
    ret = ucp_am_send_req(req, count, &ucp_ep_config(ep)->am, cb,
                          ucp_ep_config(ep)->am_u.proto);

out:
    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
    return ret;
}

static ucs_status_t 
ucp_am_handler(void *am_arg, void *am_data, size_t am_length,
               unsigned am_flags)
{
    ucp_worker_h worker = (ucp_worker_h) am_arg;
    ucp_am_hdr_t *hdr = (ucp_am_hdr_t *) am_data;
    ucs_status_t status;
    ucp_recv_desc_t *desc = NULL;
    uint16_t recv_flags = 0;
    uint16_t am_id      = hdr->am_id;
    
    if(worker->am_cbs[hdr->am_id].cb == NULL){
        ucs_warn("UCP Active Message was received with id : %u, but there" 
                  "is no registered callback for that id", hdr->am_id);
        return UCS_OK;
    }

    if(am_flags & UCT_CB_PARAM_FLAG_DESC){
        recv_flags |= UCP_RECV_DESC_FLAG_HDR;
    }
    
    status = ucp_recv_desc_init(worker, hdr + 1, am_length,
                                0, am_flags, 0,
                                recv_flags, 0, &desc);
    
    status = (worker->am_cbs[am_id].cb)(worker->am_cbs[am_id].context,
                                        desc + 1, 
                                        am_length - sizeof(ucp_am_hdr_t),
                                        UCP_CB_PARAM_FLAG_DATA);
    
    if(status == UCS_INPROGRESS && (am_flags & UCT_CB_PARAM_FLAG_DESC)) {
        status = UCS_INPROGRESS;
    } else if(status == UCS_OK && !(am_flags & UCT_CB_PARAM_FLAG_DESC)) {
        ucp_recv_desc_release(desc);
    } else {
        status = UCS_OK;
    }

    return status;
}


static ucs_status_t
ucp_am_long_handler(void *am_arg, void *am_data, size_t am_length,
               unsigned am_flags)
{ 
    ucp_worker_h worker = (ucp_worker_h) am_arg;
    ucs_status_t status;
    ucp_am_long_hdr_t *long_hdr = (ucp_am_long_hdr_t *) am_data;
    ucp_ep_h ep = ucp_worker_get_ep_by_ptr(worker, long_hdr->ep);
    ucp_ep_ext_proto_t *ep_ext = ucp_ep_ext_proto(ep);
    ucp_am_unfinished_t *parent_am;
    uint16_t am_id;

    if(worker->am_cbs[long_hdr->am_id].cb == NULL){
        ucs_warn("UCP Active Message was received with id : %u, but there"
                  "is no registered callback for that id", long_hdr->am_id);
        goto out;
    }
        
    /* if there are multiple messages,
     * we first check to see if any of the other messages
     * have arrived. If any messages have arrived,
     * we copy ourselves into the buffer and leave
     */
    ucs_list_for_each(parent_am, &ep_ext->am.started_ams, unfinished)
    {
        if (parent_am->msg_id == long_hdr->msg_id) {
            memcpy(UCS_PTR_BYTE_OFFSET(parent_am->all_data + 1, long_hdr->offset),
                   long_hdr + 1, am_length - sizeof(ucp_am_long_hdr_t));
            parent_am->left -= am_length - sizeof(ucp_am_long_hdr_t);
            /* If this is the last callback, we run it and cleanup */
            if (parent_am->left == 0) {
                am_id = long_hdr->am_id;

                status = worker->am_cbs[am_id].cb(worker->am_cbs[am_id].context,
                                                  parent_am->all_data + 1,
                                                  long_hdr->total_size,
                                                  UCP_CB_PARAM_FLAG_DATA);

                if (status != UCS_INPROGRESS) {
                    ucs_free(parent_am->all_data);
                }

                ucs_list_del(&parent_am->unfinished);
                ucs_free(parent_am);
            }
            goto out;
        }
    }
    /* If I am first, I make the buffer for everyone to go into,
     * copy myself in, and put myself on the list so people can find me
     */
    ucp_recv_desc_t *all_data = ucs_malloc(long_hdr->total_size
                                           + sizeof(ucp_recv_desc_t),
                                           "ucp recv desc for long AM");

    all_data->flags |= UCP_RECV_DESC_FLAG_MALLOC;

    size_t left = long_hdr->total_size - (am_length -
                                          sizeof(ucp_am_long_hdr_t));
    /* We know this one goes first */
    memcpy(UCS_PTR_BYTE_OFFSET(all_data + 1, long_hdr->offset),
           long_hdr + 1, am_length - sizeof(ucp_am_long_hdr_t));

    /* Can't use a desc for this because of the buffer */
    ucp_am_unfinished_t *unfinished;

    unfinished              = ucs_malloc(sizeof(ucp_am_unfinished_t),
                                         "unfinished UCP AM");
    unfinished->all_data    = all_data;
    unfinished->left        = left;
    unfinished->msg_id      = long_hdr->msg_id;

    ucs_list_add_head(&ep_ext->am.started_ams, &unfinished->unfinished);

out:
    return UCS_OK;
}



UCP_DEFINE_AM(UCP_FEATURE_AM, UCP_AM_ID_SINGLE,
              ucp_am_handler, NULL, UCT_CB_FLAG_SYNC);

UCP_DEFINE_AM(UCP_FEATURE_AM, UCP_AM_ID_MULTI,
              ucp_am_long_handler, NULL, UCT_CB_FLAG_SYNC);

const ucp_proto_t ucp_am_proto = {
    .contig_short           = ucp_am_contig_short,
    .bcopy_single           = ucp_am_bcopy_single,
    .bcopy_multi            = ucp_am_bcopy_multi,
    .zcopy_single           = ucp_am_zcopy_single,
    .zcopy_multi            = ucp_am_zcopy_multi,
    .zcopy_completion       = ucp_proto_am_zcopy_completion,
    .only_hdr_size          = sizeof(ucp_am_hdr_t),
    .first_hdr_size         = sizeof(ucp_am_long_hdr_t),
    .mid_hdr_size           = sizeof(ucp_am_long_hdr_t)
};
