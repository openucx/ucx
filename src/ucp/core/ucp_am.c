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

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_context.h>
#include <ucp/rndv/rndv.h>
#include <ucp/proto/proto_am.inl>
#include <ucp/dt/dt.h>
#include <ucp/dt/dt.inl>

#include <ucs/datastruct/array.inl>


#define UCP_AM_SHORT_REPLY_MAX_SIZE  (UCS_ALLOCA_MAX_SIZE - \
                                      sizeof(ucs_ptr_map_key_t))

UCS_ARRAY_IMPL(ucp_am_cbs, unsigned, ucp_am_entry_t, static)

ucs_status_t ucp_am_init(ucp_worker_h worker)
{
    if (!(worker->context->config.features & UCP_FEATURE_AM)) {
        return UCS_OK;
    }

    ucs_array_init_dynamic(&worker->am);
    return UCS_OK;
}

void ucp_am_cleanup(ucp_worker_h worker)
{
    if (!(worker->context->config.features & UCP_FEATURE_AM)) {
        return;
    }

    ucs_array_cleanup_dynamic(&worker->am);
}

void ucp_am_ep_init(ucp_ep_h ep)
{
    ucp_ep_ext_proto_t *ep_ext = ucp_ep_ext_proto(ep);

    if (ep->worker->context->config.features & UCP_FEATURE_AM) {
        ucs_list_head_init(&ep_ext->am.started_ams);
        ucs_queue_head_init(&ep_ext->am.mid_rdesc_q);
    }
}

void ucp_am_ep_cleanup(ucp_ep_h ep)
{
    ucp_ep_ext_proto_t *ep_ext = ucp_ep_ext_proto(ep);
    ucp_recv_desc_t *rdesc, *tmp_rdesc;
    ucs_queue_iter_t iter;
    size_t UCS_V_UNUSED count;

    if (!(ep->worker->context->config.features & UCP_FEATURE_AM)) {
        return;
    }

    count = 0;
    ucs_list_for_each_safe(rdesc, tmp_rdesc, &ep_ext->am.started_ams,
                           am_first.list) {
        ucs_list_del(&rdesc->am_first.list);
        ucs_free(rdesc);
        ++count;
    }
    ucs_trace_data("worker %p: %zu unhandled first AM fragments have been"
                   " dropped on ep %p", ep->worker, count, ep);

    count = 0;
    ucs_queue_for_each_safe(rdesc, iter, &ep_ext->am.mid_rdesc_q,
                            am_mid_queue) {
        ucs_queue_del_iter(&ep_ext->am.mid_rdesc_q, iter);
        ucp_recv_desc_release(rdesc);
        ++count;
    }
    ucs_trace_data("worker %p: %zu unhandled middle AM fragments have been"
                   " dropped on ep %p", ep->worker, count, ep);
}

size_t ucp_am_max_header_size(ucp_worker_h worker)
{
    ucp_context_h context = worker->context;
    uct_iface_attr_t *if_attr;
    ucp_rsc_index_t iface_id;
    size_t max_am_header, max_uct_fragment;
    size_t max_rts_size, max_ucp_header;

    if (!(context->config.features & UCP_FEATURE_AM)) {
        return 0ul;
    }

    max_am_header  = SIZE_MAX;
    max_rts_size   = sizeof(ucp_am_rndv_rts_hdr_t) +
                     ucp_rkey_packed_size(context, UCS_MASK(context->num_mds));
    max_ucp_header = ucs_max(max_rts_size, sizeof(ucp_am_first_hdr_t));

    /* Make sure maximal AM header can fit into one bcopy fragment
     * together with RTS or first eager header (whatever is bigger)
     */
    for (iface_id = 0; iface_id < worker->num_ifaces; ++iface_id) {
        if_attr = &worker->ifaces[iface_id]->attr;

        /* UCT_IFACE_FLAG_AM_BCOPY is required by UCP AM feature, therefore
         * at least one interface should support it.
         * Make sure that except user header single UCT fragment can fit
         * ucp_am_first_hdr_t and at least 1 byte of data. It is needed to
         * correctly use generic AM based multi-fragment protocols, which
         * expect some amount of payload to be packed to the first fragment.
         * TODO: fix generic AM based multi-fragment protocols, so that this
         * trick is not needed.
         */
        if (if_attr->cap.flags & UCT_IFACE_FLAG_AM_BCOPY) {
            max_uct_fragment = ucs_max(if_attr->cap.am.max_bcopy,
                                       max_ucp_header - 1) - max_ucp_header - 1;
            max_am_header    = ucs_min(max_am_header, max_uct_fragment);
        }
    }

    ucs_assert(max_am_header < SIZE_MAX);

    return ucs_min(max_am_header, UINT32_MAX);
}

static void ucp_am_rndv_send_ats(ucp_worker_h worker,
                                 ucp_am_rndv_rts_hdr_t *rts,
                                 ucs_status_t status)
{
    ucp_request_t *req;

    req = ucp_request_get(worker);
    if (ucs_unlikely(req == NULL)) {
        ucs_error("failed to allocate request for AM RNDV ATS");
        return;
    }

    req->send.ep = ucp_worker_get_ep_by_id(worker, rts->super.sreq.ep_id);
    req->flags   = 0;

    ucp_rndv_req_send_ats(req, NULL, rts->super.sreq.req_id, status);
}

UCS_PROFILE_FUNC_VOID(ucp_am_data_release, (worker, data),
                      ucp_worker_h worker, void *data)
{
    ucp_recv_desc_t *rdesc = (ucp_recv_desc_t *)data - 1;

    if (ucs_unlikely(rdesc->flags & UCP_RECV_DESC_FLAG_MALLOC)) {
        /* Don't use UCS_PTR_BYTE_OFFSET here due to coverity false
         * positive report. Need to step back by first_header size, where
         * originally allocated pointer resides. */
        ucs_free((char*)rdesc - sizeof(ucp_am_first_hdr_t));
        return;
    }

    if (rdesc->flags & UCP_RECV_DESC_FLAG_RNDV) {
        if (rdesc->flags & UCP_RECV_DESC_FLAG_RNDV_STARTED) {
            ucs_error("rndv receive is initiated on desc %p and cannot be released ",
                      data);
            return;
        }

        /* This data is not needed (rndv receive was not initiated), send ATS
         * back to the sender to complete its send request. */
        ucp_am_rndv_send_ats(worker, data, UCS_OK);
    }

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);
    ucp_recv_desc_release(rdesc);
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
}

static void ucp_worker_am_init_handler(ucp_worker_h worker, uint16_t id,
                                       void *context, unsigned flags,
                                       ucp_am_callback_t cb_old,
                                       ucp_am_recv_callback_t cb)
{
    ucp_am_entry_t *am_cb = &ucs_array_elem(&worker->am, id);

    am_cb->context = context;
    am_cb->flags   = flags;

    if (cb_old != NULL) {
        ucs_assert(cb == NULL);
        am_cb->cb_old = cb_old;
    } else {
        am_cb->cb     = cb;
    }
}

static ucs_status_t ucp_worker_set_am_handler_common(ucp_worker_h worker,
                                                     uint16_t id,
                                                     unsigned flags)
{
    ucs_status_t status;
    unsigned i, capacity;

    UCP_CONTEXT_CHECK_FEATURE_FLAGS(worker->context, UCP_FEATURE_AM,
                                    return UCS_ERR_INVALID_PARAM);

    if (flags >= UCP_AM_CB_PRIV_FIRST_FLAG) {
        ucs_error("unsupported flags requested for UCP AM handler: 0x%x",
                  flags);
        return UCS_ERR_INVALID_PARAM;
    }

    if (id >= ucs_array_length(&worker->am)) {
        status = ucs_array_reserve(ucp_am_cbs, &worker->am, id + 1);
        if (status != UCS_OK) {
            return status;
        }

        capacity = ucs_array_capacity(&worker->am);

        for (i = ucs_array_length(&worker->am); i < capacity; ++i) {
            ucp_worker_am_init_handler(worker, id, NULL, 0, NULL, NULL);
        }

        ucs_array_set_length(&worker->am, capacity);
    }

    return UCS_OK;
}

ucs_status_t ucp_worker_set_am_handler(ucp_worker_h worker, uint16_t id,
                                       ucp_am_callback_t cb, void *arg,
                                       uint32_t flags)
{
    ucs_status_t status;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    status = ucp_worker_set_am_handler_common(worker, id, flags);
    if (status != UCS_OK) {
        goto out;
    }

    ucp_worker_am_init_handler(worker, id, arg, flags, cb, NULL);

out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);

    return status;
}

static UCS_F_ALWAYS_INLINE int ucp_am_recv_check_id(ucp_worker_h worker,
                                                    uint16_t am_id)
{
    if (ucs_unlikely((am_id >= ucs_array_length(&worker->am)) ||
                     (ucs_array_elem(&worker->am, am_id).cb == NULL))) {
        ucs_warn("UCP Active Message was received with id : %u, but there"
                 " is no registered callback for that id", am_id);
        return 0;
    }

    return 1;
}

static UCS_F_ALWAYS_INLINE size_t
ucp_am_send_req_total_size(ucp_request_t *req)
{
    return req->send.length + req->send.msg_proto.am.header_length;
}

static UCS_F_ALWAYS_INLINE ssize_t
ucp_am_get_short_max(const ucp_request_t *req, ssize_t max_short)
{
    return (UCP_DT_IS_CONTIG(req->send.datatype) &&
            UCP_MEM_IS_ACCESSIBLE_FROM_CPU(req->send.mem_type)) ?
            max_short : -1;
}

static UCS_F_ALWAYS_INLINE void
ucp_am_fill_header(ucp_am_hdr_t *hdr, ucp_request_t *req)
{
    hdr->am_id         = req->send.msg_proto.am.am_id;
    hdr->flags         = req->send.msg_proto.am.flags;
    hdr->header_length = req->send.msg_proto.am.header_length;
}

static UCS_F_ALWAYS_INLINE void
ucp_am_fill_middle_header(ucp_am_mid_hdr_t *hdr, ucp_request_t *req)
{
    hdr->msg_id = req->send.msg_proto.message_id;
    hdr->offset = req->send.state.dt.offset;
    hdr->ep_id  = ucp_send_request_get_ep_remote_id(req);
}

static UCS_F_ALWAYS_INLINE void
ucp_am_fill_first_header(ucp_am_first_hdr_t *hdr, ucp_request_t *req)
{
    ucp_am_fill_header(&hdr->super.super, req);
    hdr->super.ep_id = ucp_send_request_get_ep_remote_id(req);
    hdr->msg_id      = req->send.msg_proto.message_id;
    hdr->total_size  = ucp_am_send_req_total_size(req);
}

static UCS_F_ALWAYS_INLINE void
ucp_am_fill_short_header(ucp_am_hdr_t *hdr, uint16_t id, uint16_t flags,
                         uint16_t header_length)
{
    UCS_STATIC_ASSERT(sizeof(*hdr) == sizeof(uint64_t));
    hdr->am_id         = id;
    hdr->flags         = flags;
    hdr->header_length = header_length;
}

static UCS_F_ALWAYS_INLINE void
ucp_am_pack_user_header(void *buffer, ucp_request_t *req)
{
    ucp_dt_state_t hdr_state;

    hdr_state.offset = 0ul;

    ucp_dt_pack(req->send.ep->worker, ucp_dt_make_contig(1),
                UCS_MEMORY_TYPE_HOST, buffer, req->send.msg_proto.am.header,
                &hdr_state, req->send.msg_proto.am.header_length);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_zcopy_pack_user_header(ucp_request_t *req)
{
    ucp_mem_desc_t *reg_desc;

    if (req->send.msg_proto.am.header_length == 0) {
        return UCS_OK;
    }

    ucs_assert(req->send.msg_proto.am.header != NULL);

    reg_desc = ucp_worker_mpool_get(&req->send.ep->worker->reg_mp);
    if (ucs_unlikely(reg_desc == NULL)) {
        return UCS_ERR_NO_MEMORY;
    }

    ucp_am_pack_user_header(reg_desc + 1, req);
    req->send.msg_proto.am.reg_desc = reg_desc;

    return UCS_OK;
}

ucs_status_t ucp_worker_set_am_recv_handler(ucp_worker_h worker,
                                            const ucp_am_handler_param_t *param)
{
    ucs_status_t status;
    uint16_t id;
    unsigned flags;

    if (!ucs_test_all_flags(param->field_mask, UCP_AM_HANDLER_PARAM_FIELD_ID |
                                               UCP_AM_HANDLER_PARAM_FIELD_CB)) {
        return UCS_ERR_INVALID_PARAM;
    }

    id    = param->id;
    flags = UCP_PARAM_VALUE(AM_HANDLER, param, flags, FLAGS, 0);

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    status = ucp_worker_set_am_handler_common(worker, id, flags);
    if (status != UCS_OK) {
        goto out;
    }

    /* cb should always be set (can be NULL) */
    ucp_worker_am_init_handler(worker, id,
                               UCP_PARAM_VALUE(AM_HANDLER, param, arg, ARG, NULL),
                               flags | UCP_AM_CB_PRIV_FLAG_NBX,
                               NULL, param->cb);

out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);

    return status;
}

static UCS_F_ALWAYS_INLINE ssize_t
ucp_am_bcopy_pack_data(void *buffer, ucp_request_t *req, size_t length)
{
    unsigned user_header_length = req->send.msg_proto.am.header_length;

    if (user_header_length != 0) {
        ucs_assert((req->send.length == 0) || (length > user_header_length));
        ucp_am_pack_user_header(buffer, req);
    }

    return user_header_length + ucp_dt_pack(req->send.ep->worker,
                                            req->send.datatype,
                                            UCS_MEMORY_TYPE_HOST,
                                            UCS_PTR_BYTE_OFFSET(buffer,
                                                            user_header_length),
                                            req->send.buffer,
                                            &req->send.state.dt,
                                            length - user_header_length);
}

static size_t
ucp_am_bcopy_pack_args_single(void *dest, void *arg)
{
    ucp_am_hdr_t *hdr  = dest;
    ucp_request_t *req = arg;
    size_t length;

    ucs_assert(req->send.state.dt.offset == 0);

    ucp_am_fill_header(hdr, req);

    length = ucp_am_bcopy_pack_data(hdr + 1, req,
                                    ucp_am_send_req_total_size(req));

    ucs_assert(length == ucp_am_send_req_total_size(req));

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

    reply_hdr->ep_id  = ucp_send_request_get_ep_remote_id(req);
    length            = ucp_am_bcopy_pack_data(reply_hdr + 1, req,
                                               ucp_am_send_req_total_size(req));

    ucs_assert(length == ucp_am_send_req_total_size(req));

    return sizeof(*reply_hdr) + length;
}

static size_t
ucp_am_bcopy_pack_args_first(void *dest, void *arg)
{
    ucp_am_first_hdr_t *hdr = dest;
    ucp_request_t *req      = arg;
    size_t length;

    length = ucs_min(ucp_am_send_req_total_size(req),
                     ucp_ep_get_max_bcopy(req->send.ep, req->send.lane) -
                     sizeof(*hdr));

    ucp_am_fill_first_header(hdr, req);

    ucs_assert(req->send.state.dt.offset == 0);

    return sizeof(*hdr) + ucp_am_bcopy_pack_data(hdr + 1, req, length);
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

    /* some amount of data should be packed in the first fragment */
    ucs_assert(req->send.state.dt.offset > 0);

    return sizeof(*hdr) + ucp_dt_pack(req->send.ep->worker, req->send.datatype,
                                      UCS_MEMORY_TYPE_HOST, hdr + 1,
                                      req->send.buffer, &req->send.state.dt,
                                      length);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_send_short(ucp_ep_h ep, uint16_t id, uint16_t flags, const void *header,
                  size_t header_length, const void *payload, size_t length)
{
    uct_ep_h am_ep = ucp_ep_get_am_uct_ep(ep);
    ucp_am_hdr_t hdr;
    void *sbuf;

    /*
     * short can't be used if both header and payload are provided
     * (to avoid packing on fast path)
     * TODO: enable short protocol for such cases when uct_am_short_iov is
     * defined in UCT
     */
    ucs_assert((length == 0ul) || (header_length == 0ul));
    ucs_assert(!(flags & UCP_AM_SEND_REPLY));
    ucp_am_fill_short_header(&hdr, id, flags, header_length);

    sbuf = (header_length != 0) ? (void*)header : (void*)payload;

    return uct_ep_am_short(am_ep, UCP_AM_ID_SINGLE, hdr.u64, sbuf,
                           length + header_length);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_send_short_reply(ucp_ep_h ep, uint16_t id, uint16_t flags,
                        const void *header, size_t header_length,
                        const void *payload, size_t length)
{
    uct_ep_h am_ep = ucp_ep_get_am_uct_ep(ep);
    size_t tx_length;
    ucp_am_hdr_t hdr;
    const void *data;
    void *tx_buffer;
    ucs_status_t status;

    ucs_assert(flags & UCP_AM_SEND_REPLY);
    ucs_assert((length == 0ul) || (header_length == 0ul));

    status = ucp_ep_resolve_remote_id(ep, ep->am_lane);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    if (header_length != 0) {
        tx_length = header_length;
        data      = header;
    } else {
        tx_length = length;
        data      = payload;
    }

    /* Reply protocol carries ep_id in its header in addition to AM short
     * header. UCT AM short protocol accepts only 8 bytes header, so add ep_id
     * right before the data.
     * TODO: Use uct_ep_am_short_iov instead, when it is defined in UCT
     */
    UCS_STATIC_ASSERT(ucs_offsetof(ucp_am_reply_hdr_t, ep_id) == sizeof(hdr));

    tx_buffer = ucs_alloca(tx_length + sizeof(ucs_ptr_map_key_t));

    *((ucs_ptr_map_key_t*)tx_buffer) = ucp_ep_remote_id(ep);

    ucp_am_fill_short_header(&hdr, id, flags, header_length);

    memcpy(UCS_PTR_BYTE_OFFSET(tx_buffer, sizeof(ucs_ptr_map_key_t)),
           data, tx_length);

    return uct_ep_am_short(am_ep, UCP_AM_ID_SINGLE_REPLY, hdr.u64, tx_buffer,
                           tx_length + sizeof(ucs_ptr_map_key_t));
}

static ucs_status_t ucp_am_contig_short(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ucs_status_t status;

    req->send.lane = ucp_ep_get_am_lane(ep);
    status         = ucp_am_send_short(ep, req->send.msg_proto.am.am_id,
                                       req->send.msg_proto.am.flags,
                                       req->send.msg_proto.am.header,
                                       req->send.msg_proto.am.header_length,
                                       req->send.buffer, req->send.length);
    if (ucs_likely(status == UCS_OK)) {
        ucp_request_complete_send(req, UCS_OK);
    }

    return status;
}

static ucs_status_t ucp_am_contig_short_reply(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ucs_status_t status;

    req->send.lane = ucp_ep_get_am_lane(ep);
    status         = ucp_am_send_short_reply(ep, req->send.msg_proto.am.am_id,
                                             req->send.msg_proto.am.flags,
                                             req->send.msg_proto.am.header,
                                             req->send.msg_proto.am.header_length,
                                             req->send.buffer, req->send.length);
    if (ucs_likely(status == UCS_OK)) {
        ucp_request_complete_send(req, UCS_OK);
    }

    return status;
}

static ucs_status_t ucp_am_bcopy_single(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_single(self, UCP_AM_ID_SINGLE,
                                                 ucp_am_bcopy_pack_args_single);

    return ucp_am_bcopy_handle_status_from_pending(self, 0, 0, status);
}

static ucs_status_t ucp_am_bcopy_single_reply(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_single(self, UCP_AM_ID_SINGLE_REPLY,
                                                 ucp_am_bcopy_pack_args_single_reply);

    return ucp_am_bcopy_handle_status_from_pending(self, 0, 0, status);
}

static ucs_status_t ucp_am_bcopy_multi(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_multi(self, UCP_AM_ID_FIRST,
                                                UCP_AM_ID_MIDDLE,
                                                ucp_am_bcopy_pack_args_first,
                                                ucp_am_bcopy_pack_args_mid, 0);

    return ucp_am_bcopy_handle_status_from_pending(self, 1, 0, status);
}

static UCS_F_ALWAYS_INLINE void ucp_am_zcopy_complete_common(ucp_request_t *req)
{
    ucs_assert(req->send.state.uct_comp.count == 0);

    if (req->send.msg_proto.am.header_length > 0) {
        ucs_mpool_put_inline(req->send.msg_proto.am.reg_desc);
    }

    ucp_request_send_buffer_dereg(req); /* TODO register+lane change */
}

static void ucp_am_zcopy_req_complete(ucp_request_t *req, ucs_status_t status)
{
    ucp_am_zcopy_complete_common(req);
    ucp_request_complete_send(req, status);
}

void ucp_am_zcopy_completion(uct_completion_t *self)
{
    ucp_request_t *req  = ucs_container_of(self, ucp_request_t,
                                           send.state.uct_comp);

    if (req->send.state.dt.offset == req->send.length) {
        ucp_am_zcopy_req_complete(req, self->status);
    } else if (self->status != UCS_OK) {
        ucs_assert(self->status != UCS_INPROGRESS);

        /* Avoid double release of resources */
        req->send.state.uct_comp.func = NULL;

        /* NOTE: the request is in pending queue if data was not completely sent,
         *       just release resources by ucp_am_zcopy_req_complete() here and
         *       complete request on purge pending later.
         * TODO: - Make sure status contains the latest error happened on the
         *         lane (to be supported in UCT)
         *       - Revise multi-rail support
         *       - Prevent other fragments to be sent
         */
        ucp_am_zcopy_complete_common(req);
    }
}

static ucs_status_t ucp_am_zcopy_single(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_am_hdr_t hdr;

    ucp_am_fill_header(&hdr, req);

    return ucp_do_am_zcopy_single(self, UCP_AM_ID_SINGLE, &hdr, sizeof(hdr),
                                  req->send.msg_proto.am.reg_desc,
                                  req->send.msg_proto.am.header_length,
                                  ucp_am_zcopy_req_complete);
}

static ucs_status_t ucp_am_zcopy_single_reply(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_am_reply_hdr_t reply_hdr;

    reply_hdr.ep_id = ucp_send_request_get_ep_remote_id(req);
    ucp_am_fill_header(&reply_hdr.super, req);

    return ucp_do_am_zcopy_single(self, UCP_AM_ID_SINGLE_REPLY, &reply_hdr,
                                  sizeof(reply_hdr),
                                  req->send.msg_proto.am.reg_desc,
                                  req->send.msg_proto.am.header_length,
                                  ucp_am_zcopy_req_complete);
}

static ucs_status_t ucp_am_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t *req       = ucs_container_of(self, ucp_request_t, send.uct);
    unsigned user_hdr_length = req->send.msg_proto.am.header_length;
    ucp_am_first_hdr_t first_hdr;
    ucp_am_mid_hdr_t mid_hdr;

    if (req->send.state.dt.offset != 0) {
        ucp_am_fill_middle_header(&mid_hdr, req);
        return ucp_do_am_zcopy_multi(self, UCP_AM_ID_FIRST, UCP_AM_ID_MIDDLE,
                                     NULL, 0ul, &mid_hdr, sizeof(mid_hdr), NULL,
                                     0ul, ucp_am_zcopy_req_complete, 1);
    }

    ucp_am_fill_first_header(&first_hdr, req);

    return ucp_do_am_zcopy_multi(self, UCP_AM_ID_FIRST, UCP_AM_ID_MIDDLE,
                                 &first_hdr, sizeof(first_hdr),
                                 NULL, sizeof(mid_hdr),
                                 req->send.msg_proto.am.reg_desc,
                                 user_hdr_length,
                                 ucp_am_zcopy_req_complete, 1);
}

size_t ucp_am_rndv_rts_pack(void *dest, void *arg)
{
    ucp_request_t *sreq               = arg;
    ucp_am_rndv_rts_hdr_t *am_rts_hdr = dest;
    size_t max_bcopy                  = ucp_ep_get_max_bcopy(sreq->send.ep,
                                                             sreq->send.lane);
    size_t rts_size, total_size;

    ucp_am_fill_header(&am_rts_hdr->am, sreq);
    rts_size = ucp_rndv_rts_pack(sreq, &am_rts_hdr->super,
                                 sizeof(*am_rts_hdr), UCP_RNDV_RTS_FLAG_AM);

    if (sreq->send.msg_proto.am.header_length == 0) {
        return rts_size;
    }

    total_size = rts_size + sreq->send.msg_proto.am.header_length;

    if (ucs_unlikely(total_size > max_bcopy)) {
        ucs_fatal("RTS is too big %lu, max %lu", total_size, max_bcopy);
    }

    ucp_am_pack_user_header(UCS_PTR_BYTE_OFFSET(am_rts_hdr, rts_size), sreq);

    return total_size;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_proto_progress_am_rndv_rts, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);
    size_t max_rts_size;

    /* RTS consists of: AM RTS header, packed rkeys and user header */
    max_rts_size = sizeof(ucp_am_rndv_rts_hdr_t) +
                   ucp_ep_config(sreq->send.ep)->rndv.rkey_size +
                   sreq->send.msg_proto.am.header_length;

    return ucp_do_am_single(self, UCP_AM_ID_RNDV_RTS, ucp_am_rndv_rts_pack,
                            max_rts_size);
}

static ucs_status_t ucp_am_send_start_rndv(ucp_request_t *sreq)
{
    ucp_trace_req(sreq, "AM start_rndv to %s buffer %p length %zu",
                  ucp_ep_peer_name(sreq->send.ep), sreq->send.buffer,
                  sreq->send.length);
    UCS_PROFILE_REQUEST_EVENT(sreq, "start_rndv", sreq->send.length);

    /* Note: no need to call ucp_ep_resolve_remote_id() here, because it
     * was done in ucp_am_send_nbx
     */
    sreq->send.uct.func = ucp_proto_progress_am_rndv_rts;
    return ucp_rndv_reg_send_buffer(sreq);
}

static void ucp_am_send_req_init(ucp_request_t *req, ucp_ep_h ep,
                                 const void *header, size_t header_length,
                                 const void *buffer, ucp_datatype_t datatype,
                                 size_t count, uint16_t flags, uint16_t am_id)
{
    req->flags                           = UCP_REQUEST_FLAG_SEND_AM;
    req->send.ep                         = ep;
    req->send.msg_proto.am.am_id         = am_id;
    req->send.msg_proto.am.flags         = flags;
    req->send.msg_proto.am.header        = (void*)header;
    req->send.msg_proto.am.header_length = header_length;
    req->send.buffer                     = (void*)buffer;
    req->send.datatype                   = datatype;
    req->send.mem_type                   = UCS_MEMORY_TYPE_HOST;
    req->send.lane                       = ep->am_lane;
    req->send.pending_lane               = UCP_NULL_LANE;

    ucp_request_send_state_init(req, datatype, count);
    req->send.length = ucp_dt_length(req->send.datatype, count,
                                     req->send.buffer, &req->send.state.dt);
}

static UCS_F_ALWAYS_INLINE size_t
ucp_am_rndv_thresh(ucp_request_t *req, const ucp_request_param_t *param,
                   ucp_ep_config_t *ep_config, uint32_t flags,
                   ssize_t *max_short)
{
    size_t rndv_rma_thresh, rndv_am_thresh;

    if (flags & UCP_AM_SEND_FLAG_EAGER) {
        return SIZE_MAX;
    } else if (flags & UCP_AM_SEND_FLAG_RNDV) {
        *max_short = -1; /* disable short, rndv is explicitly requested */
        return 0;
    } else {
        ucp_request_param_rndv_thresh(req, param, &ep_config->rndv.rma_thresh,
                                      &ep_config->rndv.am_thresh,
                                      &rndv_rma_thresh, &rndv_am_thresh);
        return ucs_min(rndv_rma_thresh, rndv_am_thresh);
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_am_send_req(ucp_request_t *req, size_t count,
                const ucp_ep_msg_config_t *msg_config,
                const ucp_request_param_t *param,
                const ucp_request_send_proto_t *proto, ssize_t max_short,
                uint32_t flags)
{
    unsigned user_header_length = req->send.msg_proto.am.header_length;
    ucp_context_t *context      = req->send.ep->worker->context;
    ucp_ep_config_t *ep_config  = ucp_ep_config(req->send.ep);
    size_t rndv_thresh;
    size_t zcopy_thresh;
    ucs_status_t status;

    if (ucs_unlikely((count != 0) && (user_header_length != 0))) {
        /*
         * TODO: Remove when/if am_short with iovs defined in UCT
         */
        max_short = -1;
    } else {
        max_short = ucp_am_get_short_max(req, max_short);
    }

    rndv_thresh = ucp_am_rndv_thresh(req, param, ep_config, flags, &max_short);

    if ((user_header_length != 0) &&
        (((user_header_length + sizeof(ucp_am_first_hdr_t) + 1) >
         context->config.ext.seg_size) || (msg_config->max_iov == 1))) {
        /*
         * If user header is specified, it will be copied to the buffer taken
         * from pre-registered memory pool. For multi-fragment zcopy protocol,
         * some amount of payload (at least 1 byte) needs to be packed to the
         * first fragment. It is needed to correctly use generic AM based
         * multi-fragment protocols, which expect some amount of payload to be
         * packed to the first fragment.
         * TODO: Consider other ways to send user header, like packing together
         * with UCT AM header, direct registration of user header buffer, etc.
         */
        zcopy_thresh = rndv_thresh;
    } else {
        zcopy_thresh = ucp_proto_get_zcopy_threshold(req, msg_config, count,
                                                     rndv_thresh);
    }

    ucs_trace_req("select am request(%p) progress algorithm datatype=0x%"PRIx64
                  " buffer=%p length=%zu header_length=%u max_short=%zd"
                  " rndv_thresh=%zu zcopy_thresh=%zu",
                  req, req->send.datatype, req->send.buffer, req->send.length,
                  req->send.msg_proto.am.header_length, max_short, rndv_thresh,
                  zcopy_thresh);

    status = ucp_request_send_start(req, max_short, zcopy_thresh, rndv_thresh,
                                    count, !!user_header_length,
                                    ucp_am_send_req_total_size(req),
                                    msg_config, proto);
    if (status != UCS_OK) {
        if (ucs_unlikely(status != UCS_ERR_NO_PROGRESS)) {
            return UCS_STATUS_PTR(status);
        }

        ucs_assert(ucp_am_send_req_total_size(req) >= rndv_thresh);

        status = ucp_am_send_start_rndv(req);
        if (status != UCS_OK) {
            return UCS_STATUS_PTR(status);
        }
    }

    if ((req->send.uct.func == proto->zcopy_single) ||
        (req->send.uct.func == proto->zcopy_multi))
    {
        status = ucp_am_zcopy_pack_user_header(req);
        if (ucs_unlikely(status != UCS_OK)) {
            return UCS_STATUS_PTR(status);
        }
    }

    /* Start the request.
     * If it is completed immediately, release the request and return the status.
     * Otherwise, return the request.
     */
    ucp_request_send(req, 0);
    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        ucp_request_imm_cmpl_param(param, req, send);
    }

    ucp_request_set_send_callback_param(param, req, send);

    return req + 1;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_try_send_short(ucp_ep_h ep, uint16_t id, uint32_t flags,
                      const void *header, size_t header_length,
                      const void *buffer, size_t length)
{
    if (ucs_unlikely(((length != 0) && (header_length != 0)) ||
                     ((ssize_t)(length + header_length) >
                      ucp_ep_config(ep)->am_u.max_eager_short)) ||
                     (flags & UCP_AM_SEND_FLAG_RNDV)) {
        goto out;
    }

    if (!(flags & UCP_AM_SEND_REPLY)) {
        return ucp_am_send_short(ep, id, flags, header, header_length,
                                 buffer, length);
    } else if ((length + header_length) < UCP_AM_SHORT_REPLY_MAX_SIZE) {
        return ucp_am_send_short_reply(ep, id, flags, header, header_length,
                                       buffer, length);
    }

out:
    return UCS_ERR_NO_RESOURCE;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_am_send_nbx,
                 (ep, id, header, header_length, buffer, count, param),
                 ucp_ep_h ep, unsigned id, const void *header,
                 size_t header_length, const void *buffer, size_t count,
                 const ucp_request_param_t *param)
{
    ucs_status_t status;
    ucs_status_ptr_t ret;
    ucp_datatype_t datatype;
    ucp_request_t *req;
    uint32_t attr_mask;
    uint32_t flags;

    UCP_CONTEXT_CHECK_FEATURE_FLAGS(ep->worker->context, UCP_FEATURE_AM,
                                    return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM));

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    flags     = ucp_request_param_flags(param);
    attr_mask = param->op_attr_mask &
                (UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FLAG_NO_IMM_CMPL);

    if (ucs_likely(attr_mask == 0)) {
        status = ucp_am_try_send_short(ep, id, flags, header, header_length,
                                       buffer, count);
        ucp_request_send_check_status(status, ret, goto out);
        datatype = ucp_dt_make_contig(1);
    } else if (attr_mask == UCP_OP_ATTR_FIELD_DATATYPE) {
        datatype = param->datatype;
        if (ucs_likely(UCP_DT_IS_CONTIG(datatype))) {
            status = ucp_am_try_send_short(ep, id, flags, header,
                                           header_length, buffer,
                                           ucp_contig_dt_length(datatype,
                                                                count));
            ucp_request_send_check_status(status, ret, goto out);
        }
    } else {
        datatype = ucp_dt_make_contig(1);
    }

    if (ucs_unlikely(param->op_attr_mask & UCP_OP_ATTR_FLAG_FORCE_IMM_CMPL)) {
        ret = UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE);
        goto out;
    }

    status = ucp_ep_resolve_remote_id(ep, ep->am_lane);
    if (ucs_unlikely(status != UCS_OK)) {
        ret = UCS_STATUS_PTR(status);
        goto out;
    }

    req = ucp_request_get_param(ep->worker, param,
                                {ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
                                 goto out;});

    ucp_am_send_req_init(req, ep, header, header_length, buffer, datatype,
                         count, flags, id);

    if (flags & UCP_AM_SEND_REPLY) {
        ret = ucp_am_send_req(req, count, &ucp_ep_config(ep)->am, param,
                              ucp_ep_config(ep)->am_u.reply_proto,
                              ucs_min(ucp_ep_config(ep)->am_u.max_eager_short,
                                      UCP_AM_SHORT_REPLY_MAX_SIZE), flags);
    } else {
        ret = ucp_am_send_req(req, count, &ucp_ep_config(ep)->am, param,
                              ucp_ep_config(ep)->am_u.proto,
                              ucp_ep_config(ep)->am_u.max_eager_short, flags);
    }

out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    return ret;
}

ucs_status_ptr_t ucp_am_send_nb(ucp_ep_h ep, uint16_t id, const void *payload,
                                size_t count, ucp_datatype_t datatype,
                                ucp_send_callback_t cb, unsigned flags)
{
    ucp_request_param_t params = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE |
                        UCP_OP_ATTR_FIELD_CALLBACK |
                        UCP_OP_ATTR_FIELD_FLAGS,
        .flags        = flags,
        .cb.send      = (ucp_send_nbx_callback_t)cb,
        .datatype     = datatype
    };

    return ucp_am_send_nbx(ep, id, NULL, 0, payload, count, &params);
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_am_recv_data_nbx,
                 (worker, data_desc, buffer, count, param),
                 ucp_worker_h worker, void *data_desc, void *buffer,
                 size_t count, const ucp_request_param_t *param)
{
    ucp_am_rndv_rts_hdr_t *rts = data_desc;
    ucp_recv_desc_t *desc      = (ucp_recv_desc_t*)data_desc - 1;
    ucs_status_ptr_t ret;
    ucp_request_t *req;
    ucp_datatype_t datatype;

    UCP_CONTEXT_CHECK_FEATURE_FLAGS(worker->context, UCP_FEATURE_AM,
                                    return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM));
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    ucs_assert(rts->super.flags & UCP_RNDV_RTS_FLAG_AM);

    if (ucs_unlikely(desc->flags & UCP_RECV_DESC_FLAG_RNDV_STARTED)) {
        ucs_error("ucp_am_recv_data_nbx was already called for desc %p",
                  data_desc);
        ret = UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM);
        goto out;
    }

    if ((count == 0ul) &&
        !(param->op_attr_mask & UCP_OP_ATTR_FLAG_NO_IMM_CMPL)) {
        ret = NULL;
        goto out;
    }

    req = ucp_request_get_param(worker, param,
                                {ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
                                 goto out;});

    /* Mark that rendezvous is started on this data descriptor */
    desc->flags       |= UCP_RECV_DESC_FLAG_RNDV_STARTED;

    /* Initialize receive request */
    datatype           = ucp_request_param_datatype(param);
    req->status        = UCS_OK;
    req->recv.worker   = worker;
    req->recv.buffer   = buffer;
    req->flags         = UCP_REQUEST_FLAG_RECV_AM;
    req->recv.datatype = datatype;
    ucp_dt_recv_state_init(&req->recv.state, buffer, datatype, count);
    req->recv.length   = ucp_dt_length(datatype, count, buffer,
                                       &req->recv.state);
    req->recv.mem_type = UCS_MEMORY_TYPE_HOST;
    req->recv.am.desc  = (ucp_recv_desc_t*)rts - 1;

    ucp_request_set_callback_param(param, recv_am, req, recv.am);

    ucs_assertv(req->recv.length >= rts->super.size,
                "rx buffer too small %zu, need %zu",
                req->recv.length, rts->super.size);

    if (count > 0ul) {
        ucp_rndv_receive(worker, req, &rts->super, rts + 1);
    } else {
        /* Nothing to receive, send ack to sender to complete its request */
        ucp_am_rndv_send_ats(worker, rts, UCS_OK);
        ucp_request_complete_am_recv(req, UCS_OK);
        desc->flags |= UCP_RECV_DESC_FLAG_COMPLETED;
    }

    ret = req + 1;

out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    return ret;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_invoke_cb(ucp_worker_h worker, ucp_am_hdr_t *am_hdr,
                 size_t am_hdr_length, size_t data_length,
                 ucp_ep_h reply_ep, uint64_t recv_flags)
{
    uint16_t           am_id = am_hdr->am_id;
    uint32_t user_hdr_length = am_hdr->header_length;
        void        *am_data = UCS_PTR_BYTE_OFFSET(am_hdr, am_hdr_length);
    ucp_am_entry_t    *am_cb = &ucs_array_elem(&worker->am, am_id);
    ucp_am_recv_param_t param;
    unsigned flags;

    if (ucs_unlikely(!ucp_am_recv_check_id(worker, am_id))) {
        return UCS_OK;
    }

    if (ucs_likely(am_cb->flags & UCP_AM_CB_PRIV_FLAG_NBX)) {
        param.recv_attr = recv_flags;
        param.reply_ep  = reply_ep;

        return am_cb->cb(am_cb->context, user_hdr_length ? am_data : NULL,
                         user_hdr_length,
                         UCS_PTR_BYTE_OFFSET(am_data, user_hdr_length),
                         data_length - user_hdr_length, &param);
    }

    if (ucs_unlikely(user_hdr_length != 0)) {
        ucs_warn("incompatible UCP Active Message routines are used, please"
                 " register handler with ucp_worker_set_am_recv_handler()\n"
                 "(or use ucp_am_send_nb() for sending)");
        return UCS_OK;
    }

    flags = (recv_flags & UCP_AM_RECV_ATTR_FLAG_DATA) ?
            UCP_CB_PARAM_FLAG_DATA : 0;

    return am_cb->cb_old(am_cb->context, am_data, data_length, reply_ep,
                         flags);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_handler_common(ucp_worker_h worker, ucp_am_hdr_t *am_hdr, size_t hdr_size,
                      size_t total_length, ucp_ep_h reply_ep, unsigned am_flags,
                      uint64_t recv_flags)
{
    ucp_recv_desc_t *desc = NULL;
    void *data;
    ucs_status_t status;

    recv_flags |= (am_flags & UCT_CB_PARAM_FLAG_DESC) ?
                  UCP_AM_RECV_ATTR_FLAG_DATA : 0;

    status      = ucp_am_invoke_cb(worker, am_hdr, hdr_size,
                                   total_length - hdr_size, reply_ep,
                                   recv_flags);
    if (status != UCS_INPROGRESS) {
        return UCS_OK; /* we do not need UCT desc, just return UCS_OK */
    }

    if (ucs_unlikely(!(am_flags & UCT_CB_PARAM_FLAG_DESC))) {
        ucs_error("can't hold data, FLAG_DATA flag is not set");
        return UCS_OK;
    }

    ucs_assert(total_length >= am_hdr->header_length + hdr_size);
    data   = UCS_PTR_BYTE_OFFSET(am_hdr, hdr_size + am_hdr->header_length);
    status = ucp_recv_desc_init(worker, data,
                                total_length - hdr_size - am_hdr->header_length,
                                0,
                                UCT_CB_PARAM_FLAG_DESC, /* pass as a const */
                                0, 0, -hdr_size, &desc);
    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        ucs_error("worker %p could not allocate descriptor for active"
                  " message on callback : %u", worker, am_hdr->am_id);
        return UCS_OK;
    }
    ucs_assert(desc != NULL);

    return UCS_INPROGRESS;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_am_handler_reply,
                 (am_arg, am_data, am_length, am_flags),
                 void *am_arg, void *am_data, size_t am_length,
                 unsigned am_flags)
{
    ucp_am_reply_hdr_t *hdr = (ucp_am_reply_hdr_t *)am_data;
    ucp_worker_h worker     = (ucp_worker_h)am_arg;
    ucp_ep_h reply_ep;

    reply_ep = UCP_WORKER_GET_EP_BY_ID(worker, hdr->ep_id, "AM (reply proto)",
                                       return UCS_OK);

    return ucp_am_handler_common(worker, &hdr->super, sizeof(*hdr),
                                 am_length, reply_ep, am_flags,
                                 UCP_AM_RECV_ATTR_FIELD_REPLY_EP);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_am_handler,
                 (am_arg, am_data, am_length, am_flags),
                 void *am_arg, void *am_data, size_t am_length,
                 unsigned am_flags)
{
    ucp_worker_h worker = am_arg;
    ucp_am_hdr_t *hdr   = am_data;

    return ucp_am_handler_common(worker, hdr, sizeof(*hdr), am_length,
                                 NULL, am_flags, 0ul);
}

static UCS_F_ALWAYS_INLINE ucp_recv_desc_t*
ucp_am_find_first_rdesc(ucp_worker_h worker, ucp_ep_ext_proto_t *ep_ext,
                       uint64_t msg_id)
{
    ucp_recv_desc_t *rdesc;
    ucp_am_first_hdr_t *first_hdr;

    ucs_list_for_each(rdesc, &ep_ext->am.started_ams, am_first.list) {
        first_hdr = (ucp_am_first_hdr_t*)(rdesc + 1);
        if (first_hdr->msg_id == msg_id) {
            return rdesc;
        }
    }

    return NULL;
}

static UCS_F_ALWAYS_INLINE void
ucp_am_copy_data_fragment(ucp_recv_desc_t *first_rdesc, void *data,
                          size_t length, size_t offset)
{
    UCS_PROFILE_NAMED_CALL("am_memcpy_recv", ucs_memcpy_relaxed,
                           UCS_PTR_BYTE_OFFSET(first_rdesc + 1, offset),
                           data, length);
    first_rdesc->am_first.remaining -= length;
}

static UCS_F_ALWAYS_INLINE uint64_t
ucp_am_hdr_reply_ep(ucp_worker_h worker, uint16_t flags, ucp_ep_h ep,
                    ucp_ep_h *reply_ep_p)
{
    if (flags & UCP_AM_SEND_REPLY) {
        *reply_ep_p = ep;
        return UCP_AM_RECV_ATTR_FIELD_REPLY_EP;
    }

    *reply_ep_p = NULL;

    return 0ul;
}

static UCS_F_ALWAYS_INLINE void
ucp_am_handle_unfinished(ucp_worker_h worker, ucp_recv_desc_t *first_rdesc,
                         void *data, size_t length, size_t offset,
                         ucp_ep_h reply_ep)
{
    ucp_am_first_hdr_t *first_hdr;
    ucs_status_t status;
    void *msg;
    uint64_t recv_flags;

    ucp_am_copy_data_fragment(first_rdesc, data, length, offset);

    if (first_rdesc->am_first.remaining > 0) {
        /* not all fragments arrived yet */
        return;
    }

    /* message assembled, remove first fragment descriptor from the list in
     * ep AM extension */
    ucs_list_del(&first_rdesc->am_first.list);

    first_hdr  = (ucp_am_first_hdr_t*)(first_rdesc + 1);
    recv_flags = ucp_am_hdr_reply_ep(worker, first_hdr->super.super.flags,
                                     reply_ep, &reply_ep);

    status     = ucp_am_invoke_cb(worker, &first_hdr->super.super,
                                  sizeof(*first_hdr), first_hdr->total_size,
                                  reply_ep,
                                  recv_flags | UCP_AM_RECV_ATTR_FLAG_DATA);
    if (status != UCS_INPROGRESS) {
        ucs_free(first_rdesc); /* user does not need to hold this data */
        return;
    }

    /* Need to reinit descriptor, because we passed data shifted by
     * ucp_am_first_hdr_t size + user header size to the cb.
     * In ucp_am_data_release function, we calculate desc as
     * "data_pointer - sizeof(desc)", which would not point to the beginning
     * of the original desc.
     * original desc layout: |desc|first_hdr|user_hdr|data|
     * new desc layout:                         |desc|data| (AM first and user
     *                                                       headers are not
     *                                                       needed anymore,
     *                                                       can overwrite)
     */
    msg                = UCS_PTR_BYTE_OFFSET(first_rdesc + 1,
                                             first_rdesc->payload_offset);
    first_rdesc        = (ucp_recv_desc_t*)msg - 1;
    first_rdesc->flags = UCP_RECV_DESC_FLAG_MALLOC;

    return;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_am_long_first_handler,
                 (am_arg, am_data, am_length, am_flags),
                 void *am_arg, void *am_data, size_t am_length,
                 unsigned am_flags)
{
    ucp_worker_h worker           = am_arg;
    ucp_am_first_hdr_t *first_hdr = am_data;
    ucp_recv_desc_t *mid_rdesc, *first_rdesc;
    ucp_ep_ext_proto_t *ep_ext;
    ucp_am_mid_hdr_t *mid_hdr;
    ucs_queue_iter_t iter;
    ucp_ep_h ep;
    size_t remaining;
    uint64_t recv_flags;

    ep        = UCP_WORKER_GET_EP_BY_ID(worker, first_hdr->super.ep_id,
                                        "AM first fragment", return UCS_OK);
    remaining = first_hdr->total_size - (am_length - sizeof(*first_hdr));

    if (ucs_unlikely(remaining == 0)) {
        /* Can be a single fragment if send was issued on stub ep */
        recv_flags = ucp_am_hdr_reply_ep(worker, first_hdr->super.super.flags,
                                         ep, &ep);

        return ucp_am_handler_common(worker, &first_hdr->super.super,
                                     sizeof(*first_hdr), am_length, ep,
                                     am_flags, recv_flags);
    }

    ep_ext = ucp_ep_ext_proto(ep);

    /* This is the first fragment, other fragments (if arrived) should be on
     * ep_ext->am.mid_rdesc_q queue */
    ucs_assert(NULL == ucp_am_find_first_rdesc(worker, ep_ext,
                                               first_hdr->msg_id));

    /* Alloc buffer for the data and its desc, as we know total_size.
     * Need to allocate a separate rdesc which would be in one contigious chunk
     * with data buffer. */
    first_rdesc = ucs_malloc(first_hdr->total_size + sizeof(ucp_recv_desc_t) +
                             sizeof(*first_hdr),
                             "ucp recv desc for long AM");
    if (ucs_unlikely(first_rdesc == NULL)) {
        ucs_error("failed to allocate buffer for assembling UCP AM (id %u)",
                  first_hdr->super.super.am_id);
        return UCS_OK; /* release UCT desc */
    }

    first_rdesc->am_first.remaining = first_hdr->total_size + sizeof(*first_hdr);
    first_rdesc->payload_offset     = sizeof(*first_hdr) +
                                      first_hdr->super.super.header_length;

    /* Copy all already arrived middle fragments to the data buffer */
    ucs_queue_for_each_safe(mid_rdesc, iter, &ep_ext->am.mid_rdesc_q,
                            am_mid_queue) {
        mid_hdr = (ucp_am_mid_hdr_t*)(mid_rdesc + 1);
        if (mid_hdr->msg_id != first_hdr->msg_id) {
            continue;
        }
        ucs_queue_del_iter(&ep_ext->am.mid_rdesc_q, iter);
        ucp_am_copy_data_fragment(first_rdesc, mid_hdr + 1,
                                  mid_rdesc->length - sizeof(*mid_hdr),
                                  mid_hdr->offset + first_rdesc->payload_offset);
        ucp_recv_desc_release(mid_rdesc);
    }

    ucs_list_add_tail(&ep_ext->am.started_ams, &first_rdesc->am_first.list);

    /* Note: copy first chunk of data together with header, which contains
     * data needed to process other fragments. */
    ucp_am_handle_unfinished(worker, first_rdesc, first_hdr, am_length, 0, ep);

    return UCS_OK; /* release UCT desc */
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_am_long_middle_handler,
                 (am_arg, am_data, am_length, am_flags),
                 void *am_arg, void *am_data, size_t am_length,
                 unsigned am_flags)
{
    ucp_worker_h worker        = am_arg;
    ucp_am_mid_hdr_t *mid_hdr  = am_data;
    uint64_t msg_id            = mid_hdr->msg_id;
    ucp_recv_desc_t *mid_rdesc = NULL, *first_rdesc = NULL;
    ucp_ep_ext_proto_t *ep_ext;
    ucp_ep_h ep;
    ucs_status_t status;

    ep          = UCP_WORKER_GET_EP_BY_ID(worker, mid_hdr->ep_id,
                                          "AM middle fragment", return UCS_OK);
    ep_ext      = ucp_ep_ext_proto(ep);
    first_rdesc = ucp_am_find_first_rdesc(worker, ep_ext, msg_id);
    if (first_rdesc != NULL) {
        /* First fragment already arrived, just copy the data */
        ucp_am_handle_unfinished(worker, first_rdesc, mid_hdr + 1,
                                 am_length - sizeof(*mid_hdr),
                                 mid_hdr->offset + first_rdesc->payload_offset,
                                 ep);
        return UCS_OK; /* data is copied, release UCT desc */
    }

    /* Init desc and put it on the queue in ep AM extension, because data
     * buffer is not allocated yet. When first fragment arrives (carrying total
     * data size), all middle fragments will be copied to the data buffer. */
    status = ucp_recv_desc_init(worker, am_data, am_length, 0, am_flags,
                                sizeof(*mid_hdr), 0, 0, &mid_rdesc);
    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        ucs_error("worker %p could not allocate desc for assembling AM",
                  worker);
        return UCS_OK; /* release UCT desc */
    }

    ucs_assert(mid_rdesc != NULL);
    ucs_queue_push(&ep_ext->am.mid_rdesc_q, &mid_rdesc->am_mid_queue);

    return status;
}

ucs_status_t ucp_am_rndv_process_rts(void *arg, void *data, size_t length,
                                     unsigned tl_flags)
{
    ucp_am_rndv_rts_hdr_t *rts = data;
    ucp_worker_h worker        = arg;
    uint16_t am_id             = rts->am.am_id;
    ucp_recv_desc_t *desc      = NULL;
    ucp_ep_h ep;
    ucp_am_entry_t *am_cb;
    ucp_am_recv_param_t param;
    ucs_status_t status, desc_status;
    void *hdr;

    ep = UCP_WORKER_GET_EP_BY_ID(worker, rts->super.sreq.ep_id, "AM RTS",
                                 { status = UCS_ERR_ENDPOINT_TIMEOUT;
                                   goto out_send_ats;
                                 });

    if (ucs_unlikely(!ucp_am_recv_check_id(worker, am_id))) {
        status = UCS_ERR_INVALID_PARAM;
        goto out_send_ats;
    }

    if (rts->am.header_length != 0) {
        ucs_assert(length >= rts->am.header_length + sizeof(*rts));
        hdr = UCS_PTR_BYTE_OFFSET(rts, length - rts->am.header_length);
    } else {
        hdr = NULL;
    }

    desc_status = ucp_recv_desc_init(worker, data, length, 0, tl_flags, 0, 0,
                                     0, &desc);
    if (ucs_unlikely(UCS_STATUS_IS_ERR(desc_status))) {
        ucs_error("worker %p could not allocate descriptor for active"
                  " message RTS on callback %u", worker, am_id);
        status = UCS_ERR_NO_MEMORY;
        goto out_send_ats;
    }

    am_cb           = &ucs_array_elem(&worker->am, am_id);
    param.recv_attr = UCP_AM_RECV_ATTR_FLAG_RNDV |
                      ucp_am_hdr_reply_ep(worker, rts->am.flags, ep,
                                          &param.reply_ep);
    status          = am_cb->cb(am_cb->context, hdr, rts->am.header_length,
                                desc + 1, rts->super.size, &param);
    if ((status == UCS_INPROGRESS) ||
        (desc->flags & UCP_RECV_DESC_FLAG_RNDV_STARTED)) {
        if (desc->flags & UCP_RECV_DESC_FLAG_COMPLETED) {
            /* User initiated rendezvous receive in the callback and it is
             * already completed. No need to save the descriptor for further use
             */
            goto out;
        }

        /* User either wants to save descriptor for later use or initiated
         * rendezvous receive (by ucp_am_recv_data_nbx) in the callback. */
        ucs_assert(!UCS_STATUS_IS_ERR(status));

        /* Set this flag after the callback invocation to distiguish the cases
         * when ucp_am_recv_data_nbx is called inside the callback or not.
         */
        desc->flags |= UCP_RECV_DESC_FLAG_RNDV;
        return desc_status;
    }

    /* User does not want to receive the data, fall through to send ATS. */

out_send_ats:
    /* Some error occured or user does not need this data. Send ATS back to the
     * sender to complete its send request. */
    ucp_am_rndv_send_ats(worker, rts, status);

out:
    if ((desc != NULL) && !(desc->flags & UCP_RECV_DESC_FLAG_UCT_DESC)) {
        /* Release descriptor if it was allocated on UCP mpool, otherwise it
         * will be freed by UCT, when UCS_OK is returned from this func. */
        ucp_recv_desc_release(desc);
    }

    return UCS_OK;
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
    .zcopy_completion       = ucp_am_zcopy_completion,
    .only_hdr_size          = sizeof(ucp_am_hdr_t)
};

const ucp_request_send_proto_t ucp_am_reply_proto = {
    .contig_short           = ucp_am_contig_short_reply,
    .bcopy_single           = ucp_am_bcopy_single_reply,
    .bcopy_multi            = ucp_am_bcopy_multi,
    .zcopy_single           = ucp_am_zcopy_single_reply,
    .zcopy_multi            = ucp_am_zcopy_multi,
    .zcopy_completion       = ucp_am_zcopy_completion,
    .only_hdr_size          = sizeof(ucp_am_reply_hdr_t)
};
