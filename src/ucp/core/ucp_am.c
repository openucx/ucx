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


ucs_status_t ucp_am_init(ucp_worker_h worker)
{
    if (!(worker->context->config.features & UCP_FEATURE_AM)) {
        return UCS_OK;
    }

    worker->am.cbs_array_len = 0ul;
    worker->am.cbs           = NULL;

    return UCS_OK;
}

void ucp_am_cleanup(ucp_worker_h worker)
{
    if (!(worker->context->config.features & UCP_FEATURE_AM)) {
        return;
    }

    ucs_free(worker->am.cbs);
    worker->am.cbs_array_len = 0;
}

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
                     " run to completion", ep->worker);
        }
    }
}

UCS_PROFILE_FUNC_VOID(ucp_am_data_release, (worker, data),
                      ucp_worker_h worker, void *data)
{
    ucp_recv_desc_t *rdesc = (ucp_recv_desc_t *)data - 1;

    if (ucs_unlikely(rdesc->flags & UCP_RECV_DESC_FLAG_MALLOC)) {
        ucs_free(rdesc);
        return;
    }

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);
    ucp_recv_desc_release(rdesc);
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_worker_set_am_handler,
                 (worker, id, cb, arg, flags),
                 ucp_worker_h worker, uint16_t id,
                 ucp_am_callback_t cb, void *arg,
                 uint32_t flags)
{
    size_t num_entries;
    ucp_am_entry_t *am_cbs;
    int i;

    UCP_CONTEXT_CHECK_FEATURE_FLAGS(worker->context, UCP_FEATURE_AM,
                                    return UCS_ERR_INVALID_PARAM);

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    if (id >= worker->am.cbs_array_len) {
        num_entries = ucs_align_up_pow2(id + 1, UCP_AM_CB_BLOCK_SIZE);
        am_cbs      = ucs_realloc(worker->am.cbs, num_entries *
                                  sizeof(ucp_am_entry_t),
                                  "UCP AM callback array");
        if (ucs_unlikely(am_cbs == NULL)) {
            ucs_error("failed to grow UCP am cbs array to %zu", num_entries);
            return UCS_ERR_NO_MEMORY;
        }

        for (i = worker->am.cbs_array_len; i < num_entries; ++i) {
            am_cbs[i].cb      = NULL;
            am_cbs[i].context = NULL;
            am_cbs[i].flags   = 0;
        }

        worker->am.cbs           = am_cbs;
        worker->am.cbs_array_len = num_entries;
    }

    worker->am.cbs[id].cb      = cb;
    worker->am.cbs[id].context = arg;
    worker->am.cbs[id].flags   = flags;

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE int ucp_am_recv_check_id(ucp_worker_h worker,
                                                    uint16_t am_id)
{
    if (ucs_unlikely((am_id >= worker->am.cbs_array_len) ||
                     (worker->am.cbs[am_id].cb == NULL))) {
        ucs_warn("UCP Active Message was received with id : %u, but there"
                 " is no registered callback for that id", am_id);
        return 0;
    }

    return 1;
}

static UCS_F_ALWAYS_INLINE void
ucp_am_fill_header(ucp_am_hdr_t *hdr, ucp_request_t *req)
{
    hdr->am_id   = req->send.msg_proto.am.am_id;
    hdr->flags   = req->send.msg_proto.am.flags;
    hdr->padding = 0;
}

static UCS_F_ALWAYS_INLINE void
ucp_am_fill_middle_header(ucp_am_mid_hdr_t *hdr, ucp_request_t *req)
{
    hdr->msg_id = req->send.msg_proto.message_id;
    hdr->offset = req->send.state.dt.offset;
    hdr->ep_ptr = ucp_request_get_dest_ep_ptr(req);
}

static UCS_F_ALWAYS_INLINE void
ucp_am_fill_first_header(ucp_am_first_hdr_t *hdr, ucp_request_t *req)
{
    ucp_am_fill_header(&hdr->super.super, req);
    hdr->super.ep_ptr = ucp_request_get_dest_ep_ptr(req);
    hdr->msg_id       = req->send.msg_proto.message_id;
    hdr->total_size   = req->send.length;
}

static size_t
ucp_am_bcopy_pack_args_single(void *dest, void *arg)
{
    ucp_am_hdr_t *hdr  = dest;
    ucp_request_t *req = arg;
    size_t length;

    ucs_assert(req->send.state.dt.offset == 0);

    ucp_am_fill_header(hdr, req);

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
    ucp_request_t *req            = arg;
    size_t length;

    ucs_assert(req->send.state.dt.offset == 0);

    ucp_am_fill_header(&reply_hdr->super, req);
    reply_hdr->ep_ptr = ucp_request_get_dest_ep_ptr(req);

    length = ucp_dt_pack(req->send.ep->worker, req->send.datatype,
                         UCS_MEMORY_TYPE_HOST, reply_hdr + 1,
                         req->send.buffer,
                         &req->send.state.dt, req->send.length);
    ucs_assert(length == req->send.length);

    return sizeof(*reply_hdr) + length;
}

static size_t
ucp_am_bcopy_pack_args_first(void *dest, void *arg)
{
    ucp_am_first_hdr_t *hdr = dest;
    ucp_request_t *req      = arg;
    size_t length;

    length = ucs_min(req->send.length,
                     ucp_ep_get_max_bcopy(req->send.ep, req->send.lane) -
                     sizeof(*hdr));

    ucp_am_fill_first_header(hdr, req);

    ucs_assert(req->send.state.dt.offset == 0);

    return sizeof(*hdr) + ucp_dt_pack(req->send.ep->worker,
                                      req->send.datatype,
                                      UCS_MEMORY_TYPE_HOST,
                                      hdr + 1, req->send.buffer,
                                      &req->send.state.dt, length);
}

static size_t
ucp_am_bcopy_pack_args_mid(void *dest, void *arg)
{
    ucp_am_mid_hdr_t *hdr = dest;
    ucp_request_t *req    = arg;
    size_t max_bcopy      = ucp_ep_get_max_bcopy(req->send.ep, req->send.lane);
    size_t length         = ucs_min(max_bcopy - sizeof(*hdr),
                                    req->send.length - req->send.state.dt.offset);

    ucp_am_fill_middle_header(hdr, req);

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

    UCS_STATIC_ASSERT(sizeof(ucp_am_hdr_t) == sizeof(uint64_t));
    hdr.am_id   = id;
    hdr.flags   = 0;
    hdr.padding = 0;

    return uct_ep_am_short(am_ep, UCP_AM_ID_SINGLE, hdr.u64,
                           (void *)payload, length);
}

static ucs_status_t ucp_am_contig_short(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ucs_status_t  status;

    req->send.lane = ucp_ep_get_am_lane(ep);
    status         = ucp_am_send_short(ep, req->send.msg_proto.am.am_id,
                                       req->send.buffer, req->send.length);
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
    ucs_status_t status = ucp_do_am_bcopy_multi(self, UCP_AM_ID_FIRST,
                                                UCP_AM_ID_MIDDLE,
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

    ucp_am_fill_header(&hdr, req);

    return ucp_do_am_zcopy_single(self, UCP_AM_ID_SINGLE, &hdr,
                                  sizeof(hdr), ucp_proto_am_zcopy_req_complete);
}

static ucs_status_t ucp_am_zcopy_single_reply(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_am_reply_hdr_t reply_hdr;

    ucp_am_fill_header(&reply_hdr.super, req);
    reply_hdr.ep_ptr = ucp_request_get_dest_ep_ptr(req);

    return ucp_do_am_zcopy_single(self, UCP_AM_ID_SINGLE_REPLY,
                                  &reply_hdr, sizeof(reply_hdr),
                                  ucp_proto_am_zcopy_req_complete);
}

static ucs_status_t ucp_am_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_am_first_hdr_t f_hdr;
    ucp_am_mid_hdr_t m_hdr;

    ucp_am_fill_first_header(&f_hdr, req);
    ucp_am_fill_middle_header(&m_hdr, req);

    return ucp_do_am_zcopy_multi(self, UCP_AM_ID_FIRST, UCP_AM_ID_MIDDLE,
                                 &f_hdr, sizeof(f_hdr), &m_hdr, sizeof(m_hdr),
                                 ucp_proto_am_zcopy_req_complete, 1);
}

static void ucp_am_send_req_init(ucp_request_t *req, ucp_ep_h ep,
                                 const void *buffer, uintptr_t datatype,
                                 size_t count, uint16_t flags,
                                 uint16_t am_id)
{
    req->flags                   = UCP_REQUEST_FLAG_SEND_AM;
    req->send.ep                 = ep;
    req->send.msg_proto.am.am_id = am_id;
    req->send.msg_proto.am.flags = flags;
    req->send.buffer             = (void *)buffer;
    req->send.datatype           = datatype;
    req->send.mem_type           = UCS_MEMORY_TYPE_HOST;
    req->send.lane               = ep->am_lane;

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

    ucp_request_set_callback(req, send.cb, (ucp_send_nbx_callback_t)cb, NULL);

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

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_handler_common(ucp_worker_h worker, void *hdr_end, size_t hdr_size,
                      size_t total_length, ucp_ep_h reply_ep, uint16_t am_id,
                      unsigned am_flags)
{
    ucp_recv_desc_t *desc = NULL;
    ucs_status_t status;
    unsigned flags;

    if (!ucp_am_recv_check_id(worker, am_id)) {
        return UCS_OK;
    }

    if (ucs_unlikely(am_flags & UCT_CB_PARAM_FLAG_DESC)) {
        flags = UCP_CB_PARAM_FLAG_DATA;
    } else {
        flags = 0;
    }

    status = worker->am.cbs[am_id].cb(worker->am.cbs[am_id].context,
                                      hdr_end, total_length - hdr_size,
                                      reply_ep, flags);
    if (status != UCS_INPROGRESS) {
        return UCS_OK; /* we do not need UCT desc, just return UCS_OK */
    }

    if (ucs_unlikely(!(flags & UCP_CB_PARAM_FLAG_DATA))) {
        ucs_error("can't hold data, UCP_CB_PARAM_FLAG_DATA flag is not set");
        return UCS_OK;
    }

    ucs_assert(am_flags & UCT_CB_PARAM_FLAG_DESC);
    status = ucp_recv_desc_init(worker, hdr_end, total_length, 0,
                                UCT_CB_PARAM_FLAG_DESC, /* pass as a const */
                                0, 0, -hdr_size, &desc);
    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        ucs_error("worker %p could not allocate descriptor for active"
                  " message on callback : %u", worker, am_id);
        return UCS_OK;
    }
    ucs_assert(desc != NULL);

    return UCS_INPROGRESS;
}

static ucs_status_t
ucp_am_handler_reply(void *am_arg, void *am_data, size_t am_length,
                     unsigned am_flags)
{
    ucp_am_reply_hdr_t *hdr = (ucp_am_reply_hdr_t *)am_data;
    ucp_worker_h worker     = (ucp_worker_h)am_arg;
    uint16_t am_id          = hdr->super.am_id;
    ucp_ep_h reply_ep;

    reply_ep = ucp_worker_get_ep_by_ptr(worker, hdr->ep_ptr);

    return ucp_am_handler_common(worker, hdr + 1, sizeof(*hdr), am_length,
                                 reply_ep, am_id, am_flags);
}

static ucs_status_t
ucp_am_handler(void *am_arg, void *am_data, size_t am_length,
               unsigned am_flags)
{
    ucp_worker_h worker = (ucp_worker_h)am_arg;
    ucp_am_hdr_t *hdr   = (ucp_am_hdr_t *)am_data;
    uint16_t am_id      = hdr->am_id;

    return ucp_am_handler_common(worker, hdr + 1, sizeof(*hdr), am_length,
                                 NULL, am_id, am_flags);
}

static UCS_F_ALWAYS_INLINE ucp_am_unfinished_t*
ucp_am_find_unfinished(ucp_worker_h worker, ucp_ep_ext_proto_t *ep_ext,
                       uint64_t msg_id)
{
    ucp_am_unfinished_t *unfinished;

    ucs_list_for_each(unfinished, &ep_ext->am.started_ams, list) {
        if (unfinished->msg_id == msg_id) {
            return unfinished;
        }
    }

    return NULL;
}

static UCS_F_ALWAYS_INLINE ucp_am_unfinished_t*
ucp_am_alloc_unfinished(ucp_worker_h worker, ucp_ep_ext_proto_t *ep_ext,
                        uint64_t msg_id)
{
    ucp_am_unfinished_t *unfinished;

    unfinished = ucs_malloc(sizeof(ucp_am_unfinished_t), "ucp multi-frag AM");
    if (ucs_unlikely(unfinished == NULL)) {
        ucs_error("failed to allocate UCP AM fragment desc");
        return NULL;
    }

    unfinished->msg_id = msg_id;
    ucs_queue_head_init(&unfinished->mid_rdesc_q);
    ucs_list_add_head(&ep_ext->am.started_ams, &unfinished->list);

    return unfinished;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_handle_unfinished(ucp_worker_h worker, ucp_am_unfinished_t *unfinished,
                         void* data, size_t length, size_t offset)
{
    uint16_t am_id;
    ucs_status_t status;

    memcpy(UCS_PTR_BYTE_OFFSET(unfinished->all_data + 1, offset), data, length);

    unfinished->left -= length;
    if (unfinished->left == 0) {
        ucs_assert(unfinished->all_data != NULL);
        ucs_assert(unfinished->total_size > 0);

        am_id = unfinished->am_id;
        if (!ucp_am_recv_check_id(worker, am_id)) {
            status = UCS_OK;
        } else {
            status = worker->am.cbs[am_id].cb(worker->am.cbs[am_id].context,
                                              unfinished->all_data + 1,
                                              unfinished->total_size,
                                              unfinished->reply_ep,
                                              UCP_CB_PARAM_FLAG_DATA);
        }

        if (status != UCS_INPROGRESS) {
            ucs_free(unfinished->all_data);
        }

        ucs_list_del(&unfinished->list); /* message assembled, remove from the list */
        ucs_free(unfinished);
    }

    return UCS_OK;
}

static ucs_status_t ucp_am_long_first_handler(void *am_arg, void *am_data,
                                              size_t am_length, unsigned am_flags)
{
    ucp_worker_h worker        = am_arg;
    ucp_am_first_hdr_t *f_hdr  = am_data;
    ucp_ep_h ep                = ucp_worker_get_ep_by_ptr(worker,
                                                          f_hdr->super.ep_ptr);
    ucp_ep_ext_proto_t *ep_ext = ucp_ep_ext_proto(ep);
    uint16_t am_id             = f_hdr->super.super.am_id;
    size_t left;
    ucp_recv_desc_t *rdesc;
    ucp_ep_h reply_ep;
    ucp_am_unfinished_t *unfinished;
    ucp_am_mid_hdr_t *m_hdr;

    reply_ep = (f_hdr->super.super.flags & UCP_AM_SEND_REPLY) ?  ep : NULL;
    left     = f_hdr->total_size - (am_length - sizeof(*f_hdr));

    if (left == 0) {
        /* Can be a single fragment if send was issued on stub ep */
        return ucp_am_handler_common(worker, f_hdr + 1, sizeof(*f_hdr),
                                     am_length, reply_ep, am_id, am_flags);
    }

    /* If there are multiple messages, we first check if any of the other
     * messages have arrived */
    unfinished = ucp_am_find_unfinished(worker, ep_ext, f_hdr->msg_id);
    if (unfinished == NULL) {
        unfinished = ucp_am_alloc_unfinished(worker, ep_ext, f_hdr->msg_id);
        if (ucs_unlikely(unfinished == NULL)) {
            return UCS_OK; /* release UCT desc */
        }
    }

    /* Alloc buffer for the data and its desc, as we know total_size */
    unfinished->all_data = ucs_malloc(f_hdr->total_size + sizeof(ucp_recv_desc_t),
                                     "ucp recv desc for long AM");
    if (ucs_unlikely(unfinished->all_data == NULL)) {
        ucs_error("failed to allocate buffer for assembling UCP AM (id %u)",
                  am_id);
        return UCS_OK; /* release UCT desc */
    }
    unfinished->all_data->flags = UCP_RECV_DESC_FLAG_MALLOC;
    unfinished->left            = f_hdr->total_size;
    unfinished->am_id           = am_id;
    unfinished->total_size      = f_hdr->total_size;
    unfinished->reply_ep        = reply_ep;

    ucs_queue_for_each_extract(rdesc, &unfinished->mid_rdesc_q, am_queue, 1) {
        m_hdr = (ucp_am_mid_hdr_t*)(rdesc + 1);
        memcpy(UCS_PTR_BYTE_OFFSET(unfinished->all_data + 1, m_hdr->offset),
               m_hdr + 1, rdesc->length - sizeof(*m_hdr));
        unfinished->left -= rdesc->length - sizeof(*m_hdr);
        ucp_recv_desc_release(rdesc);
    }

    return ucp_am_handle_unfinished(worker, unfinished, f_hdr + 1,
                                    am_length - sizeof(*f_hdr), 0);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_long_middle_handler(void *am_arg, void *am_data, size_t am_length,
                          unsigned am_flags)
{
    ucp_worker_h worker        = am_arg;
    ucp_am_mid_hdr_t *m_hdr    = am_data;
    ucp_ep_h ep                = ucp_worker_get_ep_by_ptr(worker, m_hdr->ep_ptr);
    ucp_ep_ext_proto_t *ep_ext = ucp_ep_ext_proto(ep);
    uint64_t msg_id            = m_hdr->msg_id;
    ucp_am_unfinished_t *unfinished;
    ucp_recv_desc_t *rdesc;
    ucs_status_t status;

     /* Check if some fragment arrived already */
    unfinished = ucp_am_find_unfinished(worker, ep_ext, msg_id);
    if (unfinished == NULL) {
        /* First fragment did not arrive yet. It is possible when multiple AM
         * lanes are used for sending. */
        unfinished = ucp_am_alloc_unfinished(worker, ep_ext, msg_id);
        if (ucs_unlikely(unfinished == NULL)) {
            return UCS_OK; /* release UCT desc */
        }
        unfinished->all_data   = NULL;
        unfinished->total_size = 0;
    } else if (unfinished->all_data != NULL) {
        /* First fragment already arrived, just copy the data */
        return ucp_am_handle_unfinished(worker, unfinished, m_hdr + 1,
                                        am_length - sizeof(*m_hdr),
                                        m_hdr->offset);
    }

    /* Init desc and put it on the queue in unfinished struct, because data
     * buffer is not allocated yet. When first fragment arrives (carrying total
     * data size), all middle fragments will be copied to the data buffer. */
    status = ucp_recv_desc_init(worker, am_data, am_length, 0, am_flags,
                                sizeof(*m_hdr), 0, 0, &rdesc);
    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        ucs_error("worker %p could not allocate desc for assembling AM",
                  worker);
        return UCS_OK;
    }

    ucs_assert(rdesc != NULL);
    ucs_queue_push(&unfinished->mid_rdesc_q, &rdesc->am_queue);

    return status;
}

UCP_DEFINE_AM(UCP_FEATURE_AM, UCP_AM_ID_SINGLE,
              ucp_am_handler, NULL, 0);
UCP_DEFINE_AM(UCP_FEATURE_AM, UCP_AM_ID_FIRST,
              ucp_am_long_first_handler, NULL, 0);
UCP_DEFINE_AM(UCP_FEATURE_AM, UCP_AM_ID_MIDDLE,
              ucp_am_long_middle_handler, NULL, 0);
UCP_DEFINE_AM(UCP_FEATURE_AM, UCP_AM_ID_SINGLE_REPLY,
              ucp_am_handler_reply, NULL, 0);

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
    .bcopy_multi            = ucp_am_bcopy_multi,
    .zcopy_single           = ucp_am_zcopy_single_reply,
    .zcopy_multi            = ucp_am_zcopy_multi,
    .zcopy_completion       = ucp_proto_am_zcopy_completion,
    .only_hdr_size          = sizeof(ucp_am_reply_hdr_t)
};
