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

#include <ucs/datastruct/array.inl>


UCS_ARRAY_IMPL(ucp_am_cbs, unsigned, ucp_am_entry_t, static)

ucs_status_t ucp_am_init(ucp_worker_h worker)
{
    if (!(worker->context->config.features & UCP_FEATURE_AM)) {
        return UCS_OK;
    }

    ucs_array_init_dynamic(ucp_am_cbs, &worker->am);

    return UCS_OK;
}

void ucp_am_cleanup(ucp_worker_h worker)
{
    if (!(worker->context->config.features & UCP_FEATURE_AM)) {
        return;
    }

    ucs_array_cleanup_dynamic(ucp_am_cbs, &worker->am);
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

    if (ep->worker->context->config.features & UCP_FEATURE_AM) {
        if (ucs_unlikely(!ucs_list_is_empty(&ep_ext->am.started_ams))) {
            ucs_warn("worker %p: not all UCP active messages have been"
                     " run to completion on ep %p", ep->worker, ep);
        }

        if (ucs_unlikely(!ucs_queue_is_empty(&ep_ext->am.mid_rdesc_q))) {
            ucs_warn("worker %p: unhandled middle fragments left on ep %p",
                     ep->worker, ep);
        }
    }
}

size_t ucp_am_max_header_size(ucp_worker_h worker)
{
    uct_iface_attr_t *if_attr;
    ucp_rsc_index_t iface_id;
    size_t max_am_header, max_uct_fragment;

    if (!(worker->context->config.features & UCP_FEATURE_AM)) {
        return 0ul;
    }

    max_am_header = SIZE_MAX;

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
                                       sizeof(ucp_am_first_hdr_t) - 1) -
                               sizeof(ucp_am_first_hdr_t) - 1;
            max_am_header = ucs_min(max_am_header, max_uct_fragment);
        }
    }

    ucs_assert(max_am_header < SIZE_MAX);

    return ucs_min(max_am_header, UINT32_MAX);
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
                                                    uint16_t am_id,
                                                    uint32_t user_hdr_length)
{
    if (ucs_unlikely((am_id >= ucs_array_length(&worker->am)) ||
                     (ucs_array_elem(&worker->am, am_id).cb == NULL))) {
        ucs_warn("UCP Active Message was received with id : %u, but there"
                 " is no registered callback for that id", am_id);
        return 0;
    }

    return 1;
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
    size_t user_header_length = req->send.msg_proto.am.header_length;
    ucp_dt_state_t hdr_state;

    ucs_assert((user_header_length == 0) || (length > user_header_length));

    if (user_header_length != 0) {
        hdr_state.offset = 0ul;

        ucp_dt_pack(req->send.ep->worker, ucp_dt_make_contig(1),
                    UCS_MEMORY_TYPE_HOST, buffer, req->send.msg_proto.am.header,
                    &hdr_state, user_header_length);

        req->send.length -= user_header_length;
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

    length = ucp_am_bcopy_pack_data(hdr + 1, req, req->send.length);

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
    length            = ucp_am_bcopy_pack_data(reply_hdr + 1, req,
                                               req->send.length);

    ucs_assert(length == req->send.length + req->send.msg_proto.am.header_length);

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
    ucs_assert((length == 0ul) || (header_length == 0));
    ucs_assert(!(flags & UCP_AM_SEND_REPLY));
    UCS_STATIC_ASSERT(sizeof(ucp_am_hdr_t) == sizeof(uint64_t));
    hdr.am_id         = id;
    hdr.flags         = flags;
    hdr.header_length = header_length;

    sbuf = (header_length != 0) ? (void*)header : (void*)payload;

    return uct_ep_am_short(am_ep, UCP_AM_ID_SINGLE, hdr.u64, sbuf,
                           length + header_length);
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
    ucp_am_first_hdr_t first_hdr;
    ucp_am_mid_hdr_t mid_hdr;

    ucp_am_fill_first_header(&first_hdr, req);
    ucp_am_fill_middle_header(&mid_hdr, req);

    return ucp_do_am_zcopy_multi(self, UCP_AM_ID_FIRST, UCP_AM_ID_MIDDLE,
                                 &first_hdr, sizeof(first_hdr), &mid_hdr,
                                 sizeof(mid_hdr), ucp_proto_am_zcopy_req_complete,
                                 1);
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

    ucp_request_send_state_init(req, datatype, count);
    req->send.length = header_length + ucp_dt_length(req->send.datatype, count,
                                                     req->send.buffer,
                                                     &req->send.state.dt);
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_am_send_req(ucp_request_t *req, size_t count,
                const ucp_ep_msg_config_t *msg_config,
                const ucp_request_param_t *param,
                const ucp_request_send_proto_t *proto)
{

    ssize_t max_short;
    ucs_status_t status;

    if (ucs_unlikely((count != 0) &&
                     (req->send.msg_proto.am.header_length != 0))) {
        /*
         * TODO: Remove when/if am_short with iovs defined in UCT
         */
        max_short = -1;
    } else {
        max_short = ucp_am_get_short_max(req, msg_config);
    }

    status = ucp_request_send_start(req, max_short, SIZE_MAX, SIZE_MAX,
                                    count, msg_config, proto);
    if (status != UCS_OK) {
       return UCS_STATUS_PTR(status);
    }

    /* Start the request.
     * If it is completed immediately, release the request and return the status.
     * Otherwise, return the request.
     */
    status = ucp_request_send(req, 0);
    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        ucp_request_imm_cmpl_param(param, req, status, send);
    }

    ucp_request_set_send_callback_param(param, req, send);

    return req + 1;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_try_send_short(ucp_ep_h ep, uint16_t id, uint32_t flags,
                      const void *header, size_t header_length,
                      const void *buffer, size_t length)
{
    if (ucs_likely(!(flags & UCP_AM_SEND_REPLY) &&
                   ((length == 0) || (header_length == 0)) &&
                   ((ssize_t)(length + header_length) <=
                    ucp_ep_config(ep)->am.max_short))) {
        return ucp_am_send_short(ep, id, flags, header, header_length,
                                 buffer, length);
    }

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

    status = ucp_ep_resolve_dest_ep_ptr(ep, ep->am_lane);
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
                              ucp_ep_config(ep)->am_u.reply_proto);
    } else {
        ret = ucp_am_send_req(req, count, &ucp_ep_config(ep)->am, param,
                              ucp_ep_config(ep)->am_u.proto);
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

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_am_recv_nbx,
                 (worker, data_desc, buffer, count, param),
                 ucp_worker_h worker, void *data_desc, void *buffer,
                 size_t count, const ucp_request_param_t *param)
{
    return UCS_STATUS_PTR(UCS_ERR_NOT_IMPLEMENTED);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_invoke_cb(ucp_worker_h worker, ucp_am_hdr_t *am_hdr,
                 size_t am_hdr_length, size_t data_length,
                 ucp_ep_h reply_ep, int has_desc)
{
    uint16_t           am_id = am_hdr->am_id;
    uint32_t user_hdr_length = am_hdr->header_length;
        void        *am_data = UCS_PTR_BYTE_OFFSET(am_hdr, am_hdr_length);
    ucp_am_entry_t    *am_cb = &ucs_array_elem(&worker->am, am_id);
    ucp_am_recv_param_t param;

    if (ucs_unlikely(!ucp_am_recv_check_id(worker, am_id, user_hdr_length))) {
        return UCS_OK;
    }

    if (ucs_likely(am_cb->flags & UCP_AM_CB_PRIV_FLAG_NBX)) {
        user_hdr_length = am_hdr->header_length;

        if (reply_ep != NULL) {
            param.recv_attr = UCP_AM_RECV_ATTR_FIELD_REPLY_EP;
            param.reply_ep  = reply_ep;
        } else {
            param.recv_attr = 0ul;
            param.reply_ep  = NULL;
        }

        if (has_desc) {
            param.recv_attr |= UCP_AM_RECV_ATTR_FLAG_DATA;
        }

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

    return am_cb->cb_old(am_cb->context, am_data, data_length, reply_ep,
                         has_desc ? UCP_CB_PARAM_FLAG_DATA : 0);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_handler_common(ucp_worker_h worker, ucp_am_hdr_t *am_hdr, size_t hdr_size,
                      size_t total_length, ucp_ep_h reply_ep, unsigned am_flags)
{
    ucp_recv_desc_t *desc = NULL;
    void *data;
    ucs_status_t status;

    status = ucp_am_invoke_cb(worker, am_hdr, hdr_size, total_length - hdr_size,
                              reply_ep, am_flags & UCT_CB_PARAM_FLAG_DESC);
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

static ucs_status_t
ucp_am_handler_reply(void *am_arg, void *am_data, size_t am_length,
                     unsigned am_flags)
{
    ucp_am_reply_hdr_t *hdr = (ucp_am_reply_hdr_t *)am_data;
    ucp_worker_h worker     = (ucp_worker_h)am_arg;
    ucp_ep_h reply_ep;

    reply_ep = ucp_worker_get_ep_by_ptr(worker, hdr->ep_ptr);

    return ucp_am_handler_common(worker, &hdr->super, sizeof(*hdr),
                                 am_length, reply_ep, am_flags);
}

static ucs_status_t
ucp_am_handler(void *am_arg, void *am_data, size_t am_length, unsigned am_flags)
{
    ucp_worker_h worker = am_arg;
    ucp_am_hdr_t *hdr   = am_data;

    return ucp_am_handler_common(worker, hdr, sizeof(*hdr), am_length,
                                 NULL, am_flags);
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
    memcpy(UCS_PTR_BYTE_OFFSET(first_rdesc + 1, offset), data, length);
    first_rdesc->am_first.remaining -= length;
}

static UCS_F_ALWAYS_INLINE void
ucp_am_handle_unfinished(ucp_worker_h worker, ucp_recv_desc_t *first_rdesc,
                         void *data, size_t length, size_t offset)
{
    ucp_am_first_hdr_t *first_hdr;
    ucs_status_t status;
    ucp_ep_h reply_ep;
    void *msg;

    ucp_am_copy_data_fragment(first_rdesc, data, length, offset);

    if (first_rdesc->am_first.remaining > 0) {
        /* not all fragments arrived yet */
        return;
    }

    /* message assembled, remove first fragment descriptor from the list in
     * ep AM extension */
    ucs_list_del(&first_rdesc->am_first.list);

    first_hdr = (ucp_am_first_hdr_t*)(first_rdesc + 1);
    reply_ep  = (first_hdr->super.super.flags & UCP_AM_SEND_REPLY)        ?
                ucp_worker_get_ep_by_ptr(worker, first_hdr->super.ep_ptr) : NULL;

    status    = ucp_am_invoke_cb(worker, &first_hdr->super.super,
                                 sizeof(*first_hdr), first_hdr->total_size,
                                 reply_ep, 1);
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

static ucs_status_t ucp_am_long_first_handler(void *am_arg, void *am_data,
                                              size_t am_length, unsigned am_flags)
{
    ucp_worker_h worker           = am_arg;
    ucp_am_first_hdr_t *first_hdr = am_data;
    ucp_ep_h ep                   = ucp_worker_get_ep_by_ptr(worker,
                                                             first_hdr->super.ep_ptr);
    ucp_ep_ext_proto_t *ep_ext    = ucp_ep_ext_proto(ep);
    ucp_recv_desc_t *mid_rdesc, *first_rdesc;
    ucp_ep_h reply_ep;
    ucp_am_mid_hdr_t *mid_hdr;
    ucs_queue_iter_t iter;
    size_t remaining;

    remaining = first_hdr->total_size - (am_length - sizeof(*first_hdr));

    if (ucs_unlikely(remaining == 0)) {
        /* Can be a single fragment if send was issued on stub ep */
        reply_ep = (first_hdr->super.super.flags & UCP_AM_SEND_REPLY) ?
                   ep : NULL;
        return ucp_am_handler_common(worker, &first_hdr->super.super,
                                     sizeof(*first_hdr), am_length, reply_ep,
                                     am_flags);
    }

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
                                  mid_hdr->offset + sizeof(*first_hdr));
        ucp_recv_desc_release(mid_rdesc);
    }

    ucs_list_add_tail(&ep_ext->am.started_ams, &first_rdesc->am_first.list);

    /* Note: copy first chunk of data together with header, which contains
     * data needed to process other fragments. */
    ucp_am_handle_unfinished(worker, first_rdesc, first_hdr, am_length, 0);

    return UCS_OK; /* release UCT desc */
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_long_middle_handler(void *am_arg, void *am_data, size_t am_length,
                           unsigned am_flags)
{
    ucp_worker_h worker        = am_arg;
    ucp_am_mid_hdr_t *mid_hdr  = am_data;
    ucp_ep_h ep                = ucp_worker_get_ep_by_ptr(worker,
                                                          mid_hdr->ep_ptr);
    ucp_ep_ext_proto_t *ep_ext = ucp_ep_ext_proto(ep);
    uint64_t msg_id            = mid_hdr->msg_id;
    ucp_recv_desc_t *mid_rdesc = NULL, *first_rdesc = NULL;
    ucs_status_t status;

    first_rdesc = ucp_am_find_first_rdesc(worker, ep_ext, msg_id);
    if (first_rdesc != NULL) {
        /* First fragment already arrived, just copy the data */
        ucp_am_handle_unfinished(worker, first_rdesc, mid_hdr + 1,
                                 am_length - sizeof(*mid_hdr),
                                 mid_hdr->offset + first_rdesc->payload_offset);
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
