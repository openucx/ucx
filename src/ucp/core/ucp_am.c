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
#include <ucp/rndv/rndv.inl>
#include <ucp/proto/proto_am.inl>
#include <ucp/proto/proto_common.inl>
#include <ucp/dt/dt.h>
#include <ucp/dt/dt.inl>

#include <ucs/datastruct/array.inl>


#define UCP_AM_FIRST_FRAG_META_LEN \
    (sizeof(ucp_am_hdr_t) + sizeof(ucp_am_first_ftr_t))

#define UCP_AM_MID_FRAG_META_LEN \
    (sizeof(ucp_am_hdr_t) + sizeof(ucp_am_mid_ftr_t))


UCS_ARRAY_IMPL(ucp_am_cbs, unsigned, ucp_am_entry_t, static)

ucs_status_t ucp_am_init(ucp_worker_h worker)
{
    if (!(worker->context->config.features & UCP_FEATURE_AM)) {
        return UCS_OK;
    }

    ucs_array_init_dynamic(&worker->am.cbs);
    return UCS_OK;
}

void ucp_am_cleanup(ucp_worker_h worker)
{
    if (!(worker->context->config.features & UCP_FEATURE_AM)) {
        return;
    }

    ucs_array_cleanup_dynamic(&worker->am.cbs);
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
    max_rts_size   = sizeof(ucp_rndv_rts_hdr_t) +
                     ucp_rkey_packed_size(context, UCS_MASK(context->num_mds),
                                          UCS_SYS_DEVICE_ID_UNKNOWN, 0);
    max_ucp_header = ucs_max(max_rts_size, UCP_AM_FIRST_FRAG_META_LEN);

    /* Make sure maximal AM header can fit into one bcopy fragment
     * together with RTS or first eager header (whatever is bigger)
     */
    for (iface_id = 0; iface_id < worker->num_ifaces; ++iface_id) {
        if_attr = &worker->ifaces[iface_id]->attr;

        /* UCT_IFACE_FLAG_AM_BCOPY is required by UCP AM feature, therefore
         * at least one interface should support it.
         * Make sure that except user header single UCT fragment can fit
         * first fragment header and footer and at least 1 byte of data. It is
         * needed to correctly use generic AM based multi-fragment protocols,
         * which expect some amount of payload to be packed to the first
         * fragment.
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

static void ucp_am_rndv_send_ats(ucp_worker_h worker, ucp_rndv_rts_hdr_t *rts,
                                 ucs_status_t status)
{
    ucp_request_t *req;
    ucp_ep_h ep;

    UCP_WORKER_GET_EP_BY_ID(&ep, worker, rts->sreq.ep_id, return,
                            "AM RNDV ATS");
    req = ucp_request_get(worker);
    if (ucs_unlikely(req == NULL)) {
        ucs_error("failed to allocate request for AM RNDV ATS");
        return;
    }

    req->send.ep = ep;
    req->flags   = 0;

    ucp_rndv_req_send_ack(req, rts->size, rts->sreq.req_id, status,
                          UCP_AM_ID_RNDV_ATS, "send_ats");
}

static UCS_F_ALWAYS_INLINE void ucp_am_release_long_desc(ucp_recv_desc_t *desc)
{
    /* Don't use UCS_PTR_BYTE_OFFSET here due to coverity false positive report.
     * Need to step back by release_desc_offset, where originally allocated
     * pointer resides. */
    ucs_free((char*)desc - desc->release_desc_offset);
}

static UCS_F_ALWAYS_INLINE int
ucp_am_rdesc_in_progress(ucp_recv_desc_t *desc, ucs_status_t am_cb_status)
{
    if (!(desc->flags & UCP_RECV_DESC_FLAG_AM_CB_INPROGRESS)) {
        /* Inprogress flag is cleared - it means ucp_am_recv_data_nbx operation
         * was initiated and already completed. Thus, no need to save this data
         * descriptor.
         */
        ucs_assert(desc->flags & UCP_RECV_DESC_FLAG_RECV_STARTED);
        return 0;
    } else if ((am_cb_status != UCS_INPROGRESS) &&
               (!(desc->flags & UCP_RECV_DESC_FLAG_RECV_STARTED))) {
        /* User returned UCS_OK or error (which is allowed in RNDV flow), and
         * did not initiate receive operation. Thus, according to API, this data
         * descriptor is not needed.
         */
        return 0;
    }

    return 1;
}

UCS_PROFILE_FUNC_VOID(ucp_am_data_release, (worker, data),
                      ucp_worker_h worker, void *data)
{
    ucp_recv_desc_t *rdesc = (ucp_recv_desc_t *)data - 1;

    if (ucs_unlikely(rdesc->flags & UCP_RECV_DESC_FLAG_MALLOC)) {
        ucp_am_release_long_desc(rdesc);
        return;
    }

    if (rdesc->flags & UCP_RECV_DESC_FLAG_RNDV) {
        if (rdesc->flags & UCP_RECV_DESC_FLAG_RECV_STARTED) {
            ucs_error("rndv receive is initiated on desc %p and cannot be "
                      "released ",
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
    ucp_am_entry_t *am_cb = &ucs_array_elem(&worker->am.cbs, id);

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

    if (id >= ucs_array_length(&worker->am.cbs)) {
        status = ucs_array_reserve(ucp_am_cbs, &worker->am.cbs, id + 1);
        if (status != UCS_OK) {
            return status;
        }

        capacity = ucs_array_capacity(&worker->am.cbs);

        for (i = ucs_array_length(&worker->am.cbs); i < capacity; ++i) {
            ucp_worker_am_init_handler(worker, id, NULL, 0, NULL, NULL);
        }

        ucs_array_set_length(&worker->am.cbs, capacity);
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
    if (ucs_unlikely((am_id >= ucs_array_length(&worker->am.cbs)) ||
                     (ucs_array_elem(&worker->am.cbs, am_id).cb == NULL))) {
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
            UCP_MEM_IS_HOST(req->send.mem_type)) ?
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
    UCS_STATIC_ASSERT(sizeof(*hdr) == sizeof(ucp_am_hdr_t));

    hdr->offset = req->send.state.dt.offset;
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
ucp_am_fill_middle_footer(ucp_am_mid_ftr_t *ftr, ucp_request_t *req)
{
    ftr->msg_id = req->send.msg_proto.message_id;
    ftr->ep_id  = ucp_send_request_get_ep_remote_id(req);
}

static UCS_F_ALWAYS_INLINE void
ucp_am_fill_first_footer(ucp_am_first_ftr_t *ftr, ucp_request_t *req)
{
    ucp_am_fill_middle_footer(&ftr->super, req);
    ftr->total_size = req->send.length;
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

    reg_desc = ucp_worker_mpool_get(&req->send.ep->worker->reg_mp);
    if (ucs_unlikely(reg_desc == NULL)) {
        return UCS_ERR_NO_MEMORY;
    }

    if (req->send.msg_proto.am.header_length != 0) {
        ucs_assert(req->send.msg_proto.am.header != NULL);
        ucp_am_pack_user_header(reg_desc + 1, req);
    }

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
    size_t payload_length       = length - user_header_length;
    void *user_hdr;

    ucs_assertv((req->send.length == 0) || (length > user_header_length),
                "length %zu, user_header length %u", length,
                user_header_length);


    if (user_header_length != 0) {
        /* Pack user header to the end of message/fragment */
        user_hdr = UCS_PTR_BYTE_OFFSET(buffer, payload_length);
        ucp_am_pack_user_header(user_hdr, req);
    }

    return user_header_length +
           ucp_dt_pack(req->send.ep->worker, req->send.datatype,
                       req->send.mem_type, buffer, req->send.buffer,
                       &req->send.state.dt, payload_length);
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
    ucp_am_hdr_t *hdr  = dest;
    ucp_request_t *req = arg;
    ucp_am_reply_ftr_t *ftr;
    size_t length;

    ucs_assert(req->send.state.dt.offset == 0);

    ucp_am_fill_header(hdr, req);

    length     = ucp_am_bcopy_pack_data(hdr + 1, req,
                                        ucp_am_send_req_total_size(req));
    ftr        = UCS_PTR_BYTE_OFFSET(hdr + 1, length);
    ftr->ep_id = ucp_send_request_get_ep_remote_id(req);

    ucs_assert(length == ucp_am_send_req_total_size(req));

    return sizeof(*hdr) + length + sizeof(*ftr);
}

static size_t
ucp_am_bcopy_pack_args_first(void *dest, void *arg)
{
    ucp_am_hdr_t *hdr  = dest;
    ucp_request_t *req = arg;
    ucp_am_first_ftr_t *first_ftr;
    size_t length, max_length;

    ucs_assert(req->send.state.dt.offset == 0);

    ucp_am_fill_header(hdr, req);

    max_length = ucs_min(ucp_am_send_req_total_size(req),
                         ucp_ep_get_max_bcopy(req->send.ep, req->send.lane) -
                                 UCP_AM_FIRST_FRAG_META_LEN);
    length     = ucp_am_bcopy_pack_data(hdr + 1, req, max_length);
    first_ftr  = UCS_PTR_BYTE_OFFSET(hdr + 1, length);

    ucp_am_fill_first_footer(first_ftr, req);

    return UCP_AM_FIRST_FRAG_META_LEN + length;
}

static size_t
ucp_am_bcopy_pack_args_mid(void *dest, void *arg)
{
    ucp_am_mid_hdr_t *hdr = dest;
    ucp_request_t *req    = arg;
    size_t max_bcopy      = ucp_ep_get_max_bcopy(req->send.ep, req->send.lane);
    size_t length, max_length;
    ucp_am_mid_ftr_t *mid_ftr;

    /* some amount of data should be packed in the first fragment */
    ucs_assert(req->send.state.dt.offset > 0);
    ucs_assert(max_bcopy > UCP_AM_MID_FRAG_META_LEN);

    hdr->offset = req->send.state.dt.offset;
    max_length  = ucs_min(max_bcopy - UCP_AM_MID_FRAG_META_LEN,
                          req->send.length - req->send.state.dt.offset);
    length      = ucp_dt_pack(req->send.ep->worker, req->send.datatype,
                              req->send.mem_type, hdr + 1, req->send.buffer,
                              &req->send.state.dt, max_length);
    mid_ftr     = UCS_PTR_BYTE_OFFSET(hdr + 1, length);

    ucp_am_fill_middle_footer(mid_ftr, req);

    return UCP_AM_MID_FRAG_META_LEN + length;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_send_short(ucp_ep_h ep, uint16_t id, uint16_t flags, const void *header,
                  size_t header_length, const void *payload, size_t length,
                  int is_reply)
{
    size_t iov_cnt = 0ul;
    uct_iov_t iov[4];
    uint8_t am_id;
    ucp_am_hdr_t am_hdr;
    ucp_am_reply_ftr_t ftr;
    ucs_status_t status;

    ucp_am_fill_short_header(&am_hdr, id, flags, header_length);

    ucp_add_uct_iov_elem(iov, &am_hdr, sizeof(am_hdr), UCT_MEM_HANDLE_NULL,
                         &iov_cnt);
    ucp_add_uct_iov_elem(iov, (void*)payload, length, UCT_MEM_HANDLE_NULL,
                         &iov_cnt);

    if (header_length != 0) {
        ucp_add_uct_iov_elem(iov, (void*)header, header_length,
                             UCT_MEM_HANDLE_NULL, &iov_cnt);
    }

    if (is_reply) {
        status = ucp_ep_resolve_remote_id(ep, ep->am_lane);
        if (ucs_unlikely(status != UCS_OK)) {
            return status;
        }

        am_id     = UCP_AM_ID_AM_SINGLE_REPLY;
        ftr.ep_id = ucp_ep_remote_id(ep);
        ucp_add_uct_iov_elem(iov, &ftr, sizeof(ftr), UCT_MEM_HANDLE_NULL,
                             &iov_cnt);
    } else {
        am_id = UCP_AM_ID_AM_SINGLE;
    }

    return uct_ep_am_short_iov(ucp_ep_get_am_uct_ep(ep), am_id, iov, iov_cnt);
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
                                       req->send.buffer, req->send.length, 0);
    return ucp_am_short_handle_status_from_pending(req, status);
}

static ucs_status_t ucp_am_contig_short_reply(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ucs_status_t status;

    req->send.lane = ucp_ep_get_am_lane(ep);
    status         = ucp_am_send_short(ep, req->send.msg_proto.am.am_id,
                                       req->send.msg_proto.am.flags,
                                       req->send.msg_proto.am.header,
                                       req->send.msg_proto.am.header_length,
                                       req->send.buffer, req->send.length, 1);
    return ucp_am_short_handle_status_from_pending(req, status);
}

static ucs_status_t ucp_am_bcopy_single(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_single(self, UCP_AM_ID_AM_SINGLE,
                                                 ucp_am_bcopy_pack_args_single);

    return ucp_am_bcopy_handle_status_from_pending(self, 0, 0, status);
}

static ucs_status_t ucp_am_bcopy_single_reply(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_single(self, UCP_AM_ID_AM_SINGLE_REPLY,
                                                 ucp_am_bcopy_pack_args_single_reply);

    return ucp_am_bcopy_handle_status_from_pending(self, 0, 0, status);
}

static ucs_status_t ucp_am_bcopy_multi(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_multi(self, UCP_AM_ID_AM_FIRST,
                                                UCP_AM_ID_AM_MIDDLE,
                                                ucp_am_bcopy_pack_args_first,
                                                ucp_am_bcopy_pack_args_mid, 1);

    return ucp_am_bcopy_handle_status_from_pending(self, 1, 0, status);
}

static UCS_F_ALWAYS_INLINE void ucp_am_zcopy_complete_common(ucp_request_t *req)
{
    ucs_assert(req->send.state.uct_comp.count == 0);

    ucs_mpool_put_inline(req->send.msg_proto.am.reg_desc);
    ucp_request_send_buffer_dereg(req); /* TODO register+lane change */
}

static void ucp_am_zcopy_req_complete(ucp_request_t *req, ucs_status_t status)
{
    ucp_am_zcopy_complete_common(req);
    ucp_request_complete_send(req, status);
}

static void ucp_am_zcopy_completion(uct_completion_t *self)
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

    return ucp_do_am_zcopy_single(self, UCP_AM_ID_AM_SINGLE, &hdr, sizeof(hdr),
                                  req->send.msg_proto.am.reg_desc,
                                  req->send.msg_proto.am.header_length,
                                  ucp_am_zcopy_req_complete);
}

static ucs_status_t ucp_am_zcopy_single_reply(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_am_hdr_t hdr;
    ucp_am_reply_ftr_t *ftr;

    ucp_am_fill_header(&hdr, req);
    ucs_assert(req->send.msg_proto.am.reg_desc != NULL);

    ftr        = UCS_PTR_BYTE_OFFSET(req->send.msg_proto.am.reg_desc + 1,
                                     req->send.msg_proto.am.header_length);
    ftr->ep_id = ucp_send_request_get_ep_remote_id(req);

    return ucp_do_am_zcopy_single(self, UCP_AM_ID_AM_SINGLE_REPLY, &hdr,
                                  sizeof(hdr), req->send.msg_proto.am.reg_desc,
                                  req->send.msg_proto.am.header_length +
                                          sizeof(*ftr),
                                  ucp_am_zcopy_req_complete);
}

static ucs_status_t ucp_am_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t *req       = ucs_container_of(self, ucp_request_t, send.uct);
    unsigned user_hdr_length = req->send.msg_proto.am.header_length;
    ucp_am_hdr_t hdr;
    ucp_am_mid_hdr_t mid_hdr;
    ucp_am_first_ftr_t *first_ftr;

    /* This footer is also used for middle fragments, because it contains
     * common persistent request info (ep id and msg id)
     */
    first_ftr = UCS_PTR_BYTE_OFFSET(req->send.msg_proto.am.reg_desc + 1,
                                    req->send.msg_proto.am.header_length);
    ucp_am_fill_first_footer(first_ftr, req);

    if (req->send.state.dt.offset != 0) {
        ucp_am_fill_middle_header(&mid_hdr, req);
        return ucp_do_am_zcopy_multi(self, UCP_AM_ID_AM_FIRST,
                                     UCP_AM_ID_AM_MIDDLE, NULL, 0ul, &mid_hdr,
                                     sizeof(mid_hdr),
                                     req->send.msg_proto.am.reg_desc,
                                     sizeof(first_ftr->super), user_hdr_length,
                                     ucp_am_zcopy_req_complete, 1);
    }

    ucp_am_fill_header(&hdr, req);

    return ucp_do_am_zcopy_multi(self, UCP_AM_ID_AM_FIRST, UCP_AM_ID_AM_MIDDLE,
                                 &hdr, sizeof(hdr), NULL, 0ul,
                                 req->send.msg_proto.am.reg_desc,
                                 user_hdr_length + sizeof(*first_ftr), 0ul,
                                 ucp_am_zcopy_req_complete, 1);
}

static size_t ucp_am_rndv_rts_pack(void *dest, void *arg)
{
    ucp_request_t *sreq         = arg;
    ucp_rndv_rts_hdr_t *rts_hdr = dest;
    size_t max_bcopy            = ucp_ep_get_max_bcopy(sreq->send.ep,
                                                       sreq->send.lane);
    size_t rts_size, total_size;

    ucp_am_fill_header(ucp_am_hdr_from_rts(rts_hdr), sreq);
    rts_size = ucp_rndv_rts_pack(sreq, rts_hdr, UCP_RNDV_RTS_AM);

    if (sreq->send.msg_proto.am.header_length == 0) {
        return rts_size;
    }

    total_size = rts_size + sreq->send.msg_proto.am.header_length;

    if (ucs_unlikely(total_size > max_bcopy)) {
        ucs_fatal("RTS is too big %lu, max %lu", total_size, max_bcopy);
    }

    ucp_am_pack_user_header(UCS_PTR_BYTE_OFFSET(rts_hdr, rts_size), sreq);

    return total_size;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_proto_progress_am_rndv_rts, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *sreq = ucs_container_of(self, ucp_request_t, send.uct);

    /* RTS consists of: AM RTS header, packed rkeys and user header */
    return ucp_rndv_send_rts(sreq, ucp_am_rndv_rts_pack,
                             sizeof(ucp_rndv_rts_hdr_t) +
                                     sreq->send.msg_proto.am.header_length);
}

static ucs_status_t
ucp_am_send_start_rndv(ucp_request_t *sreq, const ucp_request_param_t *param)
{
    ucp_trace_req(sreq, "AM start_rndv to %s buffer %p length %zu",
                  ucp_ep_peer_name(sreq->send.ep), sreq->send.buffer,
                  sreq->send.length);
    UCS_PROFILE_REQUEST_EVENT(sreq, "start_rndv", sreq->send.length);

    ucp_send_request_id_alloc(sreq);

    /* Note: no need to call ucp_ep_resolve_remote_id() here, because it
     * was done in ucp_am_send_nbx
     */
    sreq->send.uct.func = ucp_proto_progress_am_rndv_rts;
    return ucp_rndv_reg_send_buffer(sreq, param);
}

static void ucp_am_send_req_init(ucp_request_t *req, ucp_ep_h ep,
                                 const void *header, size_t header_length,
                                 const void *buffer, ucp_datatype_t datatype,
                                 size_t count, uint16_t flags, uint16_t am_id,
                                 const ucp_request_param_t *param)
{
    req->flags                           = UCP_REQUEST_FLAG_SEND_AM;
    req->send.ep                         = ep;
    req->send.msg_proto.am.am_id         = am_id;
    req->send.msg_proto.am.flags         = flags;
    req->send.msg_proto.am.header        = (void*)header;
    req->send.msg_proto.am.header_length = header_length;
    req->send.buffer                     = (void*)buffer;
    req->send.datatype                   = datatype;
    req->send.lane                       = ep->am_lane;
    req->send.pending_lane               = UCP_NULL_LANE;

    ucp_request_send_state_init(req, datatype, count);
    req->send.length   = ucp_dt_length(req->send.datatype, count,
                                       req->send.buffer, &req->send.state.dt);
    req->send.mem_type = ucp_request_get_memory_type(ep->worker->context,
                                                     req->send.buffer,
                                                     req->send.length, param);
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

    max_short   = ucp_am_get_short_max(req, max_short);
    rndv_thresh = ucp_am_rndv_thresh(req, param, ep_config, flags, &max_short);

    if ((msg_config->max_iov == 1) ||
        ((user_header_length + sizeof(ucp_am_first_ftr_t)) >
         context->config.ext.seg_size)) {
        /*
         * If user header is specified, it will be copied to the buffer taken
         * from pre-registered memory pool. AM footers are also copied to
         * this buffer. Disable zcopy protocol if pre-registered buffer does not
         * fit AM footers and/or user header.
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
                                    ucp_am_send_req_total_size(req), msg_config,
                                    proto, param);
    if (status != UCS_OK) {
        if (ucs_unlikely(status != UCS_ERR_NO_PROGRESS)) {
            return UCS_STATUS_PTR(status);
        }

        ucs_assert(ucp_am_send_req_total_size(req) >= rndv_thresh);

        status = ucp_am_send_start_rndv(req, param);
        if (status != UCS_OK) {
            return UCS_STATUS_PTR(status);
        }
    }

    if ((req->send.uct.func == proto->zcopy_single) ||
        (req->send.uct.func == proto->zcopy_multi)) {
        status = ucp_am_zcopy_pack_user_header(req);
        if (ucs_unlikely(status != UCS_OK)) {
            return UCS_STATUS_PTR(status);
        }
    }

    /* Start the request.
     * If it is completed immediately, release the request and return the status.
     * Otherwise, return the request.
     */
    ucp_request_send(req);
    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        /* Coverity wrongly resolves completion callback function to
         * 'ucp_cm_client_connect_progress'*/
        /* coverity[offset_free] */
        ucp_request_imm_cmpl_param(param, req, send);
    }

    ucp_request_set_send_callback_param(param, req, send);

    return req + 1;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_try_send_short(ucp_ep_h ep, uint16_t id, uint32_t flags,
                      const void *header, size_t header_length,
                      const void *buffer, size_t length,
                      const ucp_memtype_thresh_t *max_eager_short)
{
    if (ucs_unlikely(flags & UCP_AM_SEND_FLAG_RNDV)) {
        return UCS_ERR_NO_RESOURCE;
    }

    if (ucp_proto_is_inline(ep, max_eager_short, header_length + length)) {
        return ucp_am_send_short(ep, id, flags, header, header_length, buffer,
                                 length, flags & UCP_AM_SEND_FLAG_REPLY);
    }

    return UCS_ERR_NO_RESOURCE;
}

static UCS_F_ALWAYS_INLINE uint16_t ucp_am_send_nbx_get_op_flag(uint32_t flags)
{
    if (flags & UCP_AM_SEND_FLAG_EAGER) {
        return UCP_PROTO_SELECT_OP_FLAG_AM_EAGER;
    } else if (flags & UCP_AM_SEND_FLAG_RNDV) {
        return UCP_PROTO_SELECT_OP_FLAG_AM_RNDV;
    }

    return 0;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_am_send_nbx,
                 (ep, id, header, header_length, buffer, count, param),
                 ucp_ep_h ep, unsigned id, const void *header,
                 size_t header_length, const void *buffer, size_t count,
                 const ucp_request_param_t *param)
{
    ucp_worker_h worker = ep->worker;
    ucs_status_t status;
    ucs_status_ptr_t ret;
    ucp_datatype_t datatype;
    ucp_request_t *req;
    uint32_t attr_mask;
    uint32_t flags;
    ucp_memtype_thresh_t *max_short;
    const ucp_request_send_proto_t *proto;
    size_t contig_length;
    ucp_operation_id_t op_id;

    UCP_CONTEXT_CHECK_FEATURE_FLAGS(worker->context, UCP_FEATURE_AM,
                                    return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM));
    UCP_REQUEST_CHECK_PARAM(param);

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    flags     = ucp_request_param_flags(param);
    attr_mask = param->op_attr_mask &
                (UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FLAG_NO_IMM_CMPL);

    if (flags & UCP_AM_SEND_FLAG_REPLY) {
        max_short = &ucp_ep_config(ep)->am_u.max_reply_eager_short;
        proto     = ucp_ep_config(ep)->am_u.reply_proto;
        op_id     = UCP_OP_ID_AM_SEND_REPLY;
    } else {
        max_short = &ucp_ep_config(ep)->am_u.max_eager_short;
        proto     = ucp_ep_config(ep)->am_u.proto;
        op_id     = UCP_OP_ID_AM_SEND;
    }

    if (ucs_likely(attr_mask == 0)) {
        status = ucp_am_try_send_short(ep, id, flags, header, header_length,
                                       buffer, count, max_short);
        ucp_request_send_check_status(status, ret, goto out);
        datatype      = ucp_dt_make_contig(1);
        contig_length = count;
    } else if (attr_mask == UCP_OP_ATTR_FIELD_DATATYPE) {
        datatype = param->datatype;
        if (ucs_likely(UCP_DT_IS_CONTIG(datatype))) {
            contig_length = ucp_contig_dt_length(datatype, count);
            status = ucp_am_try_send_short(ep, id, flags, header, header_length,
                                           buffer, contig_length, max_short);
            ucp_request_send_check_status(status, ret, goto out);
        } else {
            contig_length = 0ul;
        }
    } else {
        datatype      = ucp_dt_make_contig(1);
        contig_length = count;
    }

    if (ucs_unlikely(param->op_attr_mask & UCP_OP_ATTR_FLAG_FORCE_IMM_CMPL)) {
        ret = UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE);
        goto out;
    }

    /* TODO: move from common code to specific protocols (REPLY_EP, multi-Eager
     * Bcopy/Zcopy,RNDV) which use remote ID */
    status = ucp_ep_resolve_remote_id(ep, ep->am_lane);
    if (ucs_unlikely(status != UCS_OK)) {
        ret = UCS_STATUS_PTR(status);
        goto out;
    }

    req = ucp_request_get_param(worker, param,
                                {ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
                                 goto out;});

    if (worker->context->config.ext.proto_enable) {
        req->send.msg_proto.am.am_id         = id;
        req->send.msg_proto.am.flags         = flags;
        req->send.msg_proto.am.header        = (void*)header;
        req->send.msg_proto.am.header_length = header_length;
        ret = ucp_proto_request_send_op(ep, &ucp_ep_config(ep)->proto_select,
                                        UCP_WORKER_CFG_INDEX_NULL, req, op_id,
                                        buffer, count, datatype, contig_length,
                                        param, header_length,
                                        ucp_am_send_nbx_get_op_flag(flags));
    } else {
        ucp_am_send_req_init(req, ep, header, header_length, buffer, datatype,
                             count, flags, id, param);

        /* Note that max_eager_short.memtype_on is always initialized to real
         * max_short value
         */
        ret = ucp_am_send_req(req, count, &ucp_ep_config(ep)->am, param, proto,
                              max_short->memtype_on, flags);
    }

out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
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
        .flags        = flags | UCP_AM_SEND_FLAG_EAGER,
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
    ucp_recv_desc_t *desc = (ucp_recv_desc_t*)data_desc - 1;
    ucp_context_h context = worker->context;
    ucs_status_ptr_t ret;
    ucp_request_t *req;
    ucp_datatype_t datatype;
    ucs_memory_type_t mem_type;
    ucp_rndv_rts_hdr_t *rts;
    ucs_status_t status;
    size_t recv_length, rkey_length;

    /* Sanity check if the descriptor has been released */
    if (ENABLE_PARAMS_CHECK &&
        ucs_unlikely(desc->flags & UCP_RECV_DESC_FLAG_RELEASED)) {
        ucs_error("attempt to receive AM data with invalid descriptor");
        return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM);
    }

    UCP_CONTEXT_CHECK_FEATURE_FLAGS(context, UCP_FEATURE_AM,
                                    return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM));
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);


    if (ucs_unlikely(desc->flags & UCP_RECV_DESC_FLAG_RECV_STARTED)) {
        ucs_error("ucp_am_recv_data_nbx was already called for desc %p, "
                  "desc flags 0x%x",
                  data_desc, desc->flags);
        ret = UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM);
        goto out;
    }

    desc->flags |= UCP_RECV_DESC_FLAG_RECV_STARTED;
    datatype     = ucp_request_param_datatype(param);
    mem_type     = ucp_request_get_memory_type(context, buffer, desc->length,
                                               param);

    ucs_trace("AM recv %s buffer %p dt 0x%lx count %zu memtype %s",
              (desc->flags & UCP_RECV_DESC_FLAG_RNDV) ? "rndv" : "eager",
              buffer, datatype, count, ucs_memory_type_names[mem_type]);

    if (ucs_unlikely((desc->flags & UCP_RECV_DESC_FLAG_RNDV) &&
                     (count > 0ul))) {
        req = ucp_request_get_param(worker, param,
                                    {ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
                                     goto out;});

        /* Initialize receive request */
        req->status        = UCS_OK;
        req->recv.worker   = worker;
        req->recv.buffer   = buffer;
        req->flags         = UCP_REQUEST_FLAG_RECV_AM;
        req->recv.datatype = datatype;
        ucp_dt_recv_state_init(&req->recv.state, buffer, datatype, count);
        req->recv.length   = ucp_dt_length(datatype, count, buffer,
                                           &req->recv.state);
        req->recv.mem_type = mem_type;
        req->recv.am.desc  = desc;
        rts                = data_desc;

#if ENABLE_DEBUG_DATA
        req->recv.proto_rndv_config = NULL;
#endif

        status = ucp_recv_request_set_user_memh(req, param);
        if (status != UCS_OK) {
            ucp_request_put_param(param, req);
            ret = UCS_STATUS_PTR(status);
            goto out;
        }

        ucp_request_set_callback_param(param, recv_am, req, recv.am);

        ucs_assert(rts->opcode == UCP_RNDV_RTS_AM);
        ucs_assertv(req->recv.length >= rts->size,
                    "rx buffer too small %zu, need %zu", req->recv.length,
                    rts->size);

        rkey_length = desc->length - sizeof(*rts) -
                      ucp_am_hdr_from_rts(rts)->header_length;
        ucp_rndv_receive_start(worker, req, rts, rts + 1, rkey_length);
        ret = req + 1;
        goto out;
    }

    if (desc->flags & UCP_RECV_DESC_FLAG_RNDV) {
        /* Nothing to receive, send ack to sender to complete its request */
        ucp_am_rndv_send_ats(worker, data_desc, UCS_OK);
        recv_length = 0ul;
        status      = UCS_OK;
    } else {
        /* data_desc represents eager message and can be received in place
         * without initializing request */
        status      = ucp_dt_unpack_only(worker, buffer, count, datatype,
                                         mem_type, data_desc, desc->length, 1);
        recv_length = desc->length;
    }

    if (param->op_attr_mask & UCP_OP_ATTR_FLAG_NO_IMM_CMPL) {
        req = ucp_request_get_param(worker, param,
                                    {ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
                                     goto out;});
        ret         = req + 1;
        req->status = status;
        req->flags  = UCP_REQUEST_FLAG_COMPLETED;
        /* Coverity wrongly resolves completion callback function to
         * 'ucp_cm_client_connect_progress'*/
        /* coverity[offset_free] */
        ucp_request_cb_param(param, req, recv_am, recv_length);
    } else {
        if (param->op_attr_mask & UCP_OP_ATTR_FIELD_RECV_INFO) {
            *param->recv_info.length = recv_length;
        }
        ret = UCS_STATUS_PTR(status);
    }

    /* Clear this flag, because receive operation is already completed and desc
     * is not needed anymore. If receive operation was invoked from UCP AM
     * callback, UCT AM handler would release this desc (by returning UCS_OK)
     * back to UCT.
     */
    desc->flags &= ~UCP_RECV_DESC_FLAG_AM_CB_INPROGRESS;

out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    return ret;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_invoke_cb(ucp_worker_h worker, uint16_t am_id, void *user_hdr,
                 uint32_t user_hdr_length, void *data, size_t data_length,
                 ucp_ep_h reply_ep, uint64_t recv_flags)
{
    ucp_am_entry_t *am_cb = &ucs_array_elem(&worker->am.cbs, am_id);
    ucp_am_recv_param_t param;
    unsigned flags;

    if (ucs_unlikely(!ucp_am_recv_check_id(worker, am_id))) {
        return UCS_OK;
    }

    if (ucs_likely(am_cb->flags & UCP_AM_CB_PRIV_FLAG_NBX)) {
        param.recv_attr = recv_flags;
        param.reply_ep  = reply_ep;

        return am_cb->cb(am_cb->context, user_hdr, user_hdr_length, data,
                         data_length, &param);
    }

    if (ucs_unlikely(user_hdr_length != 0)) {
        ucs_warn("incompatible UCP Active Message routines are used, please"
                 " register handler with ucp_worker_set_am_recv_handler()\n"
                 "(or use ucp_am_send_nb() for sending)");
        return UCS_OK;
    }

    flags = (recv_flags & UCP_AM_RECV_ATTR_FLAG_DATA) ?
            UCP_CB_PARAM_FLAG_DATA : 0;

    return am_cb->cb_old(am_cb->context, data, data_length, reply_ep, flags);
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_am_handler_common(
        ucp_worker_h worker, ucp_am_hdr_t *am_hdr, size_t total_length,
        ucp_ep_h reply_ep, unsigned am_flags, uint64_t recv_flags,
        const char *name)
{
    ucp_recv_desc_t *desc    = NULL;
    uint16_t am_id           = am_hdr->am_id;
    uint32_t user_hdr_size   = am_hdr->header_length;
    ucp_am_entry_t *am_cb    = &ucs_array_elem(&worker->am.cbs, am_id);
    void *data               = am_hdr + 1;
    size_t data_length       = total_length -
                               (sizeof(*am_hdr) + am_hdr->header_length);
    void *user_hdr           = UCS_PTR_BYTE_OFFSET(data, data_length);
    ucs_status_t desc_status = UCS_OK;
    ucs_status_t status;

    ucs_assert(total_length >= am_hdr->header_length + sizeof(*am_hdr));

    /* Initialize desc in advance, so the user could invoke ucp_am_recv_data_nbx
     * from the AM callback directly. The only exception is inline data when
     * AM callback is registered without UCP_AM_FLAG_PERSISTENT_DATA flag.
     */
    if ((am_flags & UCT_CB_PARAM_FLAG_DESC) ||
        (am_cb->flags & UCP_AM_FLAG_PERSISTENT_DATA)) {

        /* UCT may not support AM data alignment. If unaligned data ptr is
         * provided in UCT descriptor, allocate new aligned data buffer from UCP
         * AM mpool instead of using UCT descriptor directly.
         */
        if (ucs_unlikely((uintptr_t)data % worker->am.alignment)) {
            am_flags &= ~UCT_CB_PARAM_FLAG_DESC;
        }

        /* User header can not be accessed outside the user callback, so do not
         * include it to the total descriptor length. It helps to avoid extra
         * memory copy of the user header if the message is short/inlined
         * (i.e. received without UCT_CB_PARAM_FLAG_DESC flag).
         */
        desc_status = ucp_recv_desc_init(worker, data, data_length, 0, am_flags,
                                         0, UCP_RECV_DESC_FLAG_AM_CB_INPROGRESS,
                                         -(int)sizeof(*am_hdr),
                                         worker->am.alignment, name, &desc);
        if (ucs_unlikely(UCS_STATUS_IS_ERR(desc_status))) {
            ucs_error("worker %p could not allocate descriptor for active"
                      " message on callback : %u",
                      worker, am_id);
            return UCS_OK;
        }
        data        = desc + 1;
        recv_flags |= UCP_AM_RECV_ATTR_FLAG_DATA;
    }

    status = ucp_am_invoke_cb(worker, am_id, user_hdr, user_hdr_size, data,
                              data_length, reply_ep, recv_flags);
    if (desc == NULL) {
        if (ucs_unlikely(status == UCS_INPROGRESS)) {
            ucs_error("can't hold data, FLAG_DATA flag is not set");
            return UCS_OK;
        }
        ucs_assert(status == UCS_OK);

        return UCS_OK;
    }

    ucs_assert(!UCS_STATUS_IS_ERR(status));

    if (ucp_am_rdesc_in_progress(desc, status)) {
        desc->flags &= ~UCP_RECV_DESC_FLAG_AM_CB_INPROGRESS;
        return desc_status;
    } else if (!(am_flags & UCT_CB_PARAM_FLAG_DESC)) {
        ucp_recv_desc_release(desc);
    }

    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_am_handler_reply,
                 (am_arg, am_data, am_length, am_flags),
                 void *am_arg, void *am_data, size_t am_length,
                 unsigned am_flags)
{
    ucp_am_hdr_t *hdr       = (ucp_am_hdr_t*)am_data;
    ucp_worker_h worker     = (ucp_worker_h)am_arg;
    ucp_am_reply_ftr_t *ftr = UCS_PTR_BYTE_OFFSET(am_data,
                                                  am_length - sizeof(*ftr));
    ucp_ep_h reply_ep;

    UCP_WORKER_GET_VALID_EP_BY_ID(&reply_ep, worker, ftr->ep_id, return UCS_OK,
                                  "AM (reply proto)");

    return ucp_am_handler_common(worker, hdr, am_length - sizeof(ftr), reply_ep,
                                 am_flags, UCP_AM_RECV_ATTR_FIELD_REPLY_EP,
                                 "am_handler_reply");
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_am_handler,
                 (am_arg, am_data, am_length, am_flags),
                 void *am_arg, void *am_data, size_t am_length,
                 unsigned am_flags)
{
    ucp_worker_h worker = am_arg;
    ucp_am_hdr_t *hdr   = am_data;

    return ucp_am_handler_common(worker, hdr, am_length, NULL, am_flags, 0ul,
                                 "am_handler");
}

static UCS_F_ALWAYS_INLINE ucp_recv_desc_t *
ucp_am_find_first_rdesc(ucp_worker_h worker, ucp_ep_ext_proto_t *ep_ext,
                        uint64_t msg_id)
{
    ucp_recv_desc_t *rdesc;
    ucp_am_first_ftr_t *first_ftr;

    ucs_list_for_each(rdesc, &ep_ext->am.started_ams, am_first.list) {
        first_ftr = (ucp_am_first_ftr_t*)(rdesc + 1);
        if (first_ftr->super.msg_id == msg_id) {
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
    if (flags & UCP_AM_SEND_FLAG_REPLY) {
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
    ucp_am_hdr_t *hdr;
    ucp_am_first_ftr_t *first_ftr;
    ucs_status_t status;
    void *payload, *user_hdr;
    uint64_t recv_flags;
    size_t desc_offset, user_hdr_length, total_size;
    uint16_t am_id;

    ucp_am_copy_data_fragment(first_rdesc, data, length, offset);

    if (first_rdesc->am_first.remaining > 0) {
        /* not all fragments arrived yet */
        return;
    }

    /* Message assembled, remove first fragment descriptor from the list in
     * ep AM extension */
    ucs_list_del(&first_rdesc->am_first.list);

    first_ftr       = (ucp_am_first_ftr_t*)(first_rdesc + 1);
    hdr             = (ucp_am_hdr_t*)(first_ftr + 1);
    recv_flags      = ucp_am_hdr_reply_ep(worker, hdr->flags, reply_ep,
                                          &reply_ep) |
                      UCP_AM_RECV_ATTR_FLAG_DATA;
    payload         = UCS_PTR_BYTE_OFFSET(first_rdesc + 1,
                                          first_rdesc->payload_offset);
    am_id           = hdr->am_id;
    user_hdr_length = hdr->header_length;
    total_size      = first_ftr->total_size;
    user_hdr        = UCS_PTR_BYTE_OFFSET(payload, total_size);

    /* Need to reinit descriptor, because we have two headers between rdesc and
     * the data. In ucp_am_data_release() and ucp_am_recv_data_nbx() functions,
     * we calculate desc as "data_pointer - sizeof(desc)", which would not
     * point to the beginning of the original desc. The content of the first and
     * base headers are not needed anymore, can safely overwrite them.
     *
     * original desc layout: |desc|first_ftr|base_hdr|padding|data|user_hdr|
     *
     * new desc layout:                                 |desc|data|
     */
    desc_offset                      = first_rdesc->payload_offset;
    first_rdesc                      = (ucp_recv_desc_t*)payload - 1;
    first_rdesc->flags               = UCP_RECV_DESC_FLAG_MALLOC |
                                       UCP_RECV_DESC_FLAG_AM_CB_INPROGRESS;
    first_rdesc->release_desc_offset = desc_offset;
    first_rdesc->length              = total_size;
    status                           = ucp_am_invoke_cb(worker, am_id, user_hdr,
                                                        user_hdr_length,
                                                        payload, total_size,
                                                        reply_ep, recv_flags);
    if (!ucp_am_rdesc_in_progress(first_rdesc, status)) {
        /* user does not need to hold this data */
        ucp_am_release_long_desc(first_rdesc);
    } else {
        first_rdesc->flags &= ~UCP_RECV_DESC_FLAG_AM_CB_INPROGRESS;
    }

    return;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_am_long_first_handler,
                 (am_arg, am_data, am_length, am_flags),
                 void *am_arg, void *am_data, size_t am_length,
                 unsigned am_flags)
{
    ucp_worker_h worker    = am_arg;
    ucp_am_hdr_t *hdr      = am_data;
    size_t user_hdr_length = hdr->header_length;
    ucp_recv_desc_t *mid_rdesc, *first_rdesc;
    ucp_ep_ext_proto_t *ep_ext;
    ucp_am_mid_hdr_t *mid_hdr;
    ucp_am_mid_ftr_t *mid_ftr;
    ucp_am_first_ftr_t *first_ftr;
    ucs_queue_iter_t iter;
    ucp_ep_h ep;
    size_t total_length, padding;
    uint64_t recv_flags;
    void *user_hdr;

    first_ftr = UCS_PTR_BYTE_OFFSET(am_data, am_length - sizeof(*first_ftr));

    UCP_WORKER_GET_VALID_EP_BY_ID(&ep, worker, first_ftr->super.ep_id,
                                  return UCS_OK, "AM first fragment");

    total_length = first_ftr->total_size + user_hdr_length +
                   UCP_AM_FIRST_FRAG_META_LEN;

    if (ucs_unlikely(am_length == total_length)) {
        /* Can be a single fragment if send was issued on stub ep */
        recv_flags = ucp_am_hdr_reply_ep(worker, hdr->flags, ep, &ep);

        return ucp_am_handler_common(worker, hdr,
                                     am_length - sizeof(*first_ftr), ep,
                                     am_flags, recv_flags,
                                     "am_long_first_handler");
    }

    ep_ext = ucp_ep_ext_proto(ep);

    /* This is the first fragment, other fragments (if arrived) should be on
     * ep_ext->am.mid_rdesc_q queue */
    ucs_assert(NULL == ucp_am_find_first_rdesc(worker, ep_ext,
                                               first_ftr->super.msg_id));

    /* Alloc buffer for the data and its desc, as we know total_size.
     * Need to allocate a separate rdesc which would be in one contigious chunk
     * with data buffer. The layout of assembled message is below:
     *
     * +-------+-----------+--------+---------+---------+----------+
     * | rdesc | first_ftr | am_hdr | padding | payload | user hdr |
     * +-------+-----------+--------+---------+---------+----------+
     *
     * Note: footer is added right after rdesc (unlike wire format) for easier
     * access to it while processing incoming fragments.
     */
    first_rdesc = ucs_malloc(total_length + sizeof(ucp_recv_desc_t) +
                                     worker->am.alignment,
                             "ucp recv desc for long AM");
    if (ucs_unlikely(first_rdesc == NULL)) {
        ucs_error("failed to allocate buffer for assembling UCP AM (id %u)",
                  hdr->am_id);
        return UCS_OK; /* release UCT desc */
    }

    padding = ucs_padding((uintptr_t)UCS_PTR_BYTE_OFFSET(
                                  first_rdesc + 1, UCP_AM_FIRST_FRAG_META_LEN),
                          worker->am.alignment);

    first_rdesc->payload_offset     = UCP_AM_FIRST_FRAG_META_LEN + padding;
    first_rdesc->am_first.remaining = first_ftr->total_size;

    /* Copy first fragment and base headers before the data, it will be needed
     * for middle fragments processing. */
    UCS_PROFILE_NAMED_CALL("am_memcpy_recv", ucs_memcpy_relaxed,
                           first_rdesc + 1, first_ftr, sizeof(*first_ftr));
    UCS_PROFILE_NAMED_CALL("am_memcpy_recv", ucs_memcpy_relaxed,
                           UCS_PTR_BYTE_OFFSET(first_rdesc + 1,
                                               sizeof(*first_ftr)),
                           hdr, sizeof(*hdr));

    /* Copy user header to the end of message */
    user_hdr = UCS_PTR_BYTE_OFFSET(first_ftr, -user_hdr_length);
    UCS_PROFILE_NAMED_CALL("am_memcpy_recv", ucs_memcpy_relaxed,
                           UCS_PTR_BYTE_OFFSET(first_rdesc + 1,
                                               first_rdesc->payload_offset +
                                                       first_ftr->total_size),
                           user_hdr, user_hdr_length);

    /* Copy all already arrived middle fragments to the data buffer */
    ucs_queue_for_each_safe(mid_rdesc, iter, &ep_ext->am.mid_rdesc_q,
                            am_mid_queue) {
        mid_ftr = UCS_PTR_BYTE_OFFSET(mid_rdesc + 1,
                                      mid_rdesc->length - sizeof(*mid_ftr));
        if (mid_ftr->msg_id != first_ftr->super.msg_id) {
            continue;
        }

        mid_hdr = (ucp_am_mid_hdr_t*)(mid_rdesc + 1);
        ucs_queue_del_iter(&ep_ext->am.mid_rdesc_q, iter);
        ucp_am_copy_data_fragment(first_rdesc, mid_hdr + 1,
                                  mid_rdesc->length - UCP_AM_MID_FRAG_META_LEN,
                                  mid_hdr->offset +
                                          first_rdesc->payload_offset);
        ucp_recv_desc_release(mid_rdesc);
    }

    ucs_list_add_tail(&ep_ext->am.started_ams, &first_rdesc->am_first.list);

    /* Note: copy first chunk of data together with AM header, which contains
     * data needed to process other fragments. */
    ucp_am_handle_unfinished(worker, first_rdesc, hdr + 1,
                             am_length - (user_hdr_length +
                                          UCP_AM_FIRST_FRAG_META_LEN),
                             first_rdesc->payload_offset, ep);

    return UCS_OK; /* release UCT desc */
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_am_long_middle_handler,
                 (am_arg, am_data, am_length, am_flags),
                 void *am_arg, void *am_data, size_t am_length,
                 unsigned am_flags)
{
    ucp_worker_h worker        = am_arg;
    ucp_am_mid_hdr_t *mid_hdr  = am_data;
    ucp_recv_desc_t *mid_rdesc = NULL, *first_rdesc = NULL;
    ucp_am_mid_ftr_t *mid_ftr;
    ucp_ep_ext_proto_t *ep_ext;
    ucp_ep_h ep;
    ucs_status_t status;

    ucs_assertv(am_length > UCP_AM_MID_FRAG_META_LEN,
                "%ld > %ld", am_length, UCP_AM_MID_FRAG_META_LEN);

    mid_ftr = UCS_PTR_BYTE_OFFSET(am_data, am_length - sizeof(*mid_ftr));

    UCP_WORKER_GET_VALID_EP_BY_ID(&ep, worker, mid_ftr->ep_id, return UCS_OK,
                                  "AM middle fragment");

    ep_ext      = ucp_ep_ext_proto(ep);
    first_rdesc = ucp_am_find_first_rdesc(worker, ep_ext, mid_ftr->msg_id);
    if (first_rdesc != NULL) {
        /* First fragment already arrived, just copy the data */
        ucp_am_handle_unfinished(worker, first_rdesc, mid_hdr + 1,
                                 am_length - UCP_AM_MID_FRAG_META_LEN,
                                 mid_hdr->offset + first_rdesc->payload_offset,
                                 ep);
        return UCS_OK; /* data is copied, release UCT desc */
    }

    /* Init desc and put it on the queue in ep AM extension, because data
     * buffer is not allocated yet. When first fragment arrives (carrying total
     * data size), all middle fragments will be copied to the data buffer. */
    status = ucp_recv_desc_init(worker, am_data, am_length, 0, am_flags,
                                sizeof(*mid_hdr), 0, 0, 1,
                                "am_long_middle_handler", &mid_rdesc);
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
    ucp_rndv_rts_hdr_t *rts = data;
    ucp_worker_h worker     = arg;
    ucp_am_hdr_t *am        = ucp_am_hdr_from_rts(rts);
    uint16_t am_id          = am->am_id;
    ucp_recv_desc_t *desc   = NULL;
    ucp_am_entry_t *am_cb   = &ucs_array_elem(&worker->am.cbs, am_id);
    ucp_ep_h ep;
    ucp_am_recv_param_t param;
    ucs_status_t status, desc_status;
    void *hdr;

    if (ENABLE_PARAMS_CHECK && !(am_cb->flags & UCP_AM_CB_PRIV_FLAG_NBX)) {
        ucs_error("active message callback registered with "
                  "ucp_worker_set_am_handler() API does not support rendezvous "
                  "protocol, the sender side should use ucp_am_send_nbx() API");
        status = UCS_ERR_INVALID_PARAM;
        goto out_send_ats;
    }

    UCP_WORKER_GET_VALID_EP_BY_ID(&ep, worker, rts->sreq.ep_id,
                                  { status = UCS_ERR_CANCELED;
                                     goto out_send_ats; },
                                  "AM RTS");

    if (ucs_unlikely(!ucp_am_recv_check_id(worker, am_id))) {
        status = UCS_ERR_INVALID_PARAM;
        goto out_send_ats;
    }

    if (am->header_length != 0) {
        ucs_assert(length >= am->header_length + sizeof(*rts));
        hdr = UCS_PTR_BYTE_OFFSET(rts, length - am->header_length);
    } else {
        hdr = NULL;
    }

    desc_status = ucp_recv_desc_init(worker, data, length, 0, tl_flags, 0,
                                     UCP_RECV_DESC_FLAG_RNDV |
                                     UCP_RECV_DESC_FLAG_AM_CB_INPROGRESS, 0, 1,
                                     "am_rndv_process_rts", &desc);
    if (ucs_unlikely(UCS_STATUS_IS_ERR(desc_status))) {
        ucs_error("worker %p could not allocate descriptor for active"
                  " message RTS on callback %u", worker, am_id);
        status = UCS_ERR_NO_MEMORY;
        goto out_send_ats;
    }

    param.recv_attr = UCP_AM_RECV_ATTR_FLAG_RNDV |
                      ucp_am_hdr_reply_ep(worker, am->flags, ep,
                                          &param.reply_ep);
    status          = am_cb->cb(am_cb->context, hdr, am->header_length,
                                desc + 1, rts->size, &param);
    if (ucp_am_rdesc_in_progress(desc, status)) {
        /* User either wants to save descriptor for later use or initiated
         * rendezvous receive (by ucp_am_recv_data_nbx) in the callback. */
        ucs_assertv(!UCS_STATUS_IS_ERR(status), "%s",
                    ucs_status_string(status));

        desc->flags &= ~UCP_RECV_DESC_FLAG_AM_CB_INPROGRESS;
        return desc_status;
    } else if (desc->flags & UCP_RECV_DESC_FLAG_RECV_STARTED) {
        /* User initiated rendezvous receive in the callback and it is
         * already completed. No need to save the descriptor for further use
         */
        goto out;
    }

    ucs_trace_data("worker %p, RTS is dropped, length %zu, status %s",
                   worker, length, ucs_status_string(status));

    /* User does not want to receive the data, fall through to send ATS. */

out_send_ats:
    /* Some error occurred or user does not need this data. Send ATS back to the
     * sender to complete its send request. */
    ucp_am_rndv_send_ats(worker, rts, status);

out:
    if (desc != NULL) {
        if (ENABLE_PARAMS_CHECK) {
            /* Specifying the descriptor as released. This can detect the use of
             * the invalid descriptor in the case when the user returns UCS_OK
             * from the AM callback and then wrongly tries to receive data with
             * ucp_am_recv_data_nbx(). */
            desc->flags |= UCP_RECV_DESC_FLAG_RELEASED;
        }
        if (!(desc->flags & UCP_RECV_DESC_FLAG_UCT_DESC)) {
            /* Release descriptor if it was allocated on UCP mpool, otherwise it
             * will be freed by UCT, when UCS_OK is returned from this func. */
            ucp_recv_desc_release(desc);
        }
    }

    return UCS_OK;
}

UCP_DEFINE_AM(UCP_FEATURE_AM, UCP_AM_ID_AM_SINGLE,
              ucp_am_handler, NULL, 0);
UCP_DEFINE_AM(UCP_FEATURE_AM, UCP_AM_ID_AM_FIRST,
              ucp_am_long_first_handler, NULL, 0);
UCP_DEFINE_AM(UCP_FEATURE_AM, UCP_AM_ID_AM_MIDDLE,
              ucp_am_long_middle_handler, NULL, 0);
UCP_DEFINE_AM(UCP_FEATURE_AM, UCP_AM_ID_AM_SINGLE_REPLY,
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
    .only_hdr_size          = sizeof(ucp_am_hdr_t) + sizeof(ucp_am_reply_ftr_t)
};
