/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "proto_rndv.inl"
#include "rndv_mtype.inl"

#include <ucp/proto/proto_debug.h>
#include <ucp/proto/proto_init.h>
#include <ucp/proto/proto_single.inl>


/**
 * RTR protocol callback, which is called when all incoming data is filled to
 * the request send buffer.
 *
 * @param in_buffer Whether data is already in user buffer (dt_iter), or in the
 * buffer that we published as the remote address.
 */
typedef void (*ucp_proto_rndv_rtr_data_received_cb_t)(ucp_request_t *req,
                                                      int in_buffer);

typedef struct {
    ucp_proto_rndv_ctrl_priv_t            super;
    uct_pack_callback_t                   pack_cb;
    ucp_proto_rndv_rtr_data_received_cb_t data_received;
} ucp_proto_rndv_rtr_priv_t;


static ucs_status_t
ucp_proto_rndv_rtr_common_init(const ucp_proto_init_params_t *init_params,
                               uint64_t rndv_modes, size_t max_length,
                               ucs_linear_func_t unpack_time,
                               ucp_proto_perf_node_t *unpack_perf_node,
                               ucp_md_map_t md_map,
                               ucs_memory_type_t mem_type,
                               ucs_sys_device_t sys_dev)
{
    ucp_context_h context                    = init_params->worker->context;
    ucp_proto_rndv_ctrl_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = context->config.ext.proto_overhead_rndv_rtr,
        .super.cfg_thresh    = ucp_proto_rndv_cfg_thresh(context, rndv_modes),
        .super.cfg_priority  = 0,
        .super.min_length    = 1,
        .super.max_length    = max_length,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = sizeof(ucp_rndv_rtr_hdr_t),
        .super.send_op       = UCT_EP_OP_AM_BCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RESPONSE |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .remote_op_id        = UCP_OP_ID_RNDV_SEND,
        .unpack_time         = unpack_time,
        .unpack_perf_node    = unpack_perf_node,
        .perf_bias           = 0.0,
        .mem_info.type       = mem_type,
        .mem_info.sys_dev    = sys_dev,
        .ctrl_msg_name       = UCP_PROTO_RNDV_RTR_NAME,
        .md_map              = md_map
    };
    ucs_status_t status;

    status = ucp_proto_rndv_ctrl_am_init(&params);
    if (status != UCS_OK) {
        return status;
    }

    *init_params->priv_size = sizeof(ucp_proto_rndv_rtr_priv_t);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rtr_common_request_init(ucp_request_t *req)
{
    ucp_send_request_id_alloc(req);
    req->send.state.completed_size = 0;
}

static ucs_status_t ucp_proto_rndv_rtr_common_send(ucp_request_t *req)
{
    const ucp_proto_rndv_rtr_priv_t *rpriv = req->send.proto_config->priv;
    ucp_worker_h UCS_V_UNUSED worker       = req->send.ep->worker;
    size_t max_rtr_size;
    ucs_status_t status;

    max_rtr_size = sizeof(ucp_rndv_rtr_hdr_t) + rpriv->super.packed_rkey_size;
    status       = ucp_proto_am_bcopy_single_progress(req, UCP_AM_ID_RNDV_RTR,
                                                      rpriv->super.lane,
                                                      rpriv->pack_cb, req,
                                                      max_rtr_size, NULL, 0);
    if (status == UCS_OK) {
        UCP_WORKER_STAT_RNDV(worker, RTR, +1);
    }

    return status;
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_rtr_hdr_pack(ucp_request_t *req, ucp_rndv_rtr_hdr_t *rtr,
                            void *buffer)
{
    rtr->sreq_id = req->send.rndv.remote_req_id;
    rtr->rreq_id = ucp_send_request_get_id(req);
    rtr->size    = req->send.state.dt_iter.length;
    rtr->offset  = req->send.rndv.offset;
    rtr->address = (uintptr_t)buffer;
    ucs_assert(rtr->size > 0);
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_rtr_common_complete(ucp_request_t *req, unsigned dt_mask)
{
    ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 1, dt_mask);
    if (req->send.rndv.rkey != NULL) {
        ucp_proto_rndv_rkey_destroy(req);
    }

    ucp_proto_rndv_recv_complete(req);
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_rtr_data_received(ucp_request_t *req, int in_buffer)
{
    ucp_send_request_id_release(req);
    ucp_proto_rndv_rtr_common_complete(req, UCP_DT_MASK_ALL);
}

static size_t ucp_proto_rndv_rtr_pack_without_rkey(void *dest, void *arg)
{
    ucp_rndv_rtr_hdr_t *rtr = dest;
    ucp_request_t *req      = arg;

    ucp_proto_rndv_rtr_hdr_pack(req, rtr, NULL);

    return sizeof(*rtr);
}

static size_t ucp_proto_rndv_rtr_pack_with_rkey(void *dest, void *arg)
{
    ucp_rndv_rtr_hdr_t *rtr            = dest;
    ucp_request_t *req                 = arg;
    const ucp_datatype_iter_t *dt_iter = &req->send.state.dt_iter;
    const ucp_proto_rndv_rtr_priv_t *rpriv;
    size_t rkey_size;

    rpriv = req->send.proto_config->priv;

    ucs_assert(dt_iter->dt_class == UCP_DATATYPE_CONTIG);

    ucp_proto_rndv_rtr_hdr_pack(req, rtr, dt_iter->type.contig.buffer);

    rkey_size = ucp_proto_request_pack_rkey(req, rpriv->super.md_map,
                                            rpriv->super.sys_dev_map,
                                            rpriv->super.sys_dev_distance,
                                            rtr + 1);
    ucs_assertv(rkey_size == rpriv->super.packed_rkey_size,
                "rkey_size=%zu exp=%zu", rkey_size,
                rpriv->super.packed_rkey_size);

    return sizeof(*rtr) + rkey_size;
}

static ucs_status_t ucp_proto_rndv_rtr_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    const ucp_proto_rndv_rtr_priv_t *rpriv = req->send.proto_config->priv;
    ucs_status_t status;

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        status = ucp_datatype_iter_mem_reg(req->send.ep->worker->context,
                                           &req->send.state.dt_iter,
                                           rpriv->super.md_map,
                                           UCT_MD_MEM_ACCESS_REMOTE_PUT |
                                           UCT_MD_MEM_FLAG_HIDE_ERRORS,
                                           UCP_DT_MASK_ALL);
        if (status != UCS_OK) {
            ucp_proto_request_abort(req, status);
            return UCS_OK;
        }

        ucp_proto_rtr_common_request_init(req);
        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    return ucp_proto_rndv_rtr_common_send(req);
}

static ucs_status_t
ucp_proto_rndv_rtr_init(const ucp_proto_init_params_t *init_params)
{
    const uint64_t rndv_modes        = UCS_BIT(UCP_RNDV_MODE_PUT_ZCOPY) |
                                       UCS_BIT(UCP_RNDV_MODE_AM);
    ucp_proto_rndv_rtr_priv_t *rpriv = init_params->priv;
    ucs_status_t status;

    if (!ucp_proto_rndv_op_check(init_params, UCP_OP_ID_RNDV_RECV, 0)) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = ucp_proto_rndv_rtr_common_init(init_params, rndv_modes, SIZE_MAX,
                                            UCS_LINEAR_FUNC_ZERO, NULL, 0,
                                            init_params->select_param->mem_type,
                                            init_params->select_param->sys_dev);
    if (status != UCS_OK) {
        return status;
    }

    rpriv->data_received = ucp_proto_rndv_rtr_data_received;

    if (init_params->select_param->dt_class == UCP_DATATYPE_CONTIG) {
        rpriv->pack_cb = ucp_proto_rndv_rtr_pack_with_rkey;
    } else {
        rpriv->pack_cb = ucp_proto_rndv_rtr_pack_without_rkey;
    }

    return UCS_OK;
}

static void ucp_proto_rndv_rtr_query(const ucp_proto_query_params_t *params,
                                     ucp_proto_query_attr_t *attr)
{
    const ucp_proto_rndv_ctrl_priv_t *rpriv = params->priv;

    ucp_proto_select_elem_query(params->worker, &rpriv->remote_proto,
                                params->msg_length, attr);
    attr->is_estimation = 1;
    attr->lane_map      = UCS_BIT(rpriv->lane);
}

static void ucp_proto_rndv_rtr_abort_super(void *request, ucs_status_t status,
                                           void *user_data)
{
    ucp_request_t *req = (ucp_request_t*)request - 1;
    ucp_proto_rndv_recv_complete(req);
}

static void ucp_proto_rndv_rtr_abort(ucp_request_t *req, ucs_status_t status)
{
    const ucp_proto_rndv_rtr_priv_t *rpriv = req->send.proto_config->priv;
    ucp_request_t *rreq                    = ucp_request_get_super(req);

    rreq->status = status;
    ucp_request_set_callback(req, send.cb, ucp_proto_rndv_rtr_abort_super);

    if (ucp_request_memh_invalidate(req, status)) {
        if (req->send.rndv.rkey != NULL) {
            ucp_proto_rndv_rkey_destroy(req);
        }
        ucp_proto_request_zcopy_id_reset(req);
        return;
    }

    rpriv->data_received(req, 0);
}

ucp_proto_t ucp_rndv_rtr_proto = {
    .name     = "rndv/rtr",
    .desc     = NULL,
    .flags    = 0,
    .init     = ucp_proto_rndv_rtr_init,
    .query    = ucp_proto_rndv_rtr_query,
    .progress = {ucp_proto_rndv_rtr_progress},
    .abort    = ucp_proto_rndv_rtr_abort,
    .reset    = ucp_proto_request_zcopy_id_reset
};

static size_t ucp_proto_rndv_rtr_mtype_pack(void *dest, void *arg)
{
    ucp_rndv_rtr_hdr_t *rtr                = dest;
    ucp_request_t *req                     = arg;
    const ucp_proto_rndv_rtr_priv_t *rpriv = req->send.proto_config->priv;
    ucp_md_map_t md_map                    = rpriv->super.md_map;
    ucp_mem_desc_t *mdesc                  = req->send.rndv.mdesc;
    ucp_memory_info_t mem_info;
    ssize_t packed_rkey_size;

    ucs_assert(mdesc != NULL);
    ucp_proto_rndv_rtr_hdr_pack(req, rtr, mdesc->ptr);

    ucs_assert(ucs_test_all_flags(mdesc->memh->md_map, md_map));

    /* Pack remote key for the fragment */
    mem_info.type    = mdesc->memh->mem_type;
    mem_info.sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
    packed_rkey_size = ucp_rkey_pack_memh(req->send.ep->worker->context, md_map,
                                          mdesc->memh, mdesc->ptr,
                                          req->send.state.dt_iter.length,
                                          &mem_info, 0, NULL, 0, rtr + 1);
    if (packed_rkey_size < 0) {
        ucs_error("failed to pack remote key: %s",
                  ucs_status_string((ucs_status_t)packed_rkey_size));
        packed_rkey_size = 0;
    }

    return sizeof(*rtr) + packed_rkey_size;
}

static void ucp_proto_rndv_rtr_mtype_complete(ucp_request_t *req)
{
    ucs_mpool_put_inline(req->send.rndv.mdesc);
    if (ucp_proto_rndv_request_is_ppln_frag(req)) {
        ucp_proto_rndv_ppln_recv_frag_complete(req, 0, 0);
    } else {
        ucp_proto_rndv_rtr_common_complete(req, UCS_BIT(UCP_DATATYPE_CONTIG));
    }
}

static void
ucp_proto_rndv_rtr_mtype_abort(ucp_request_t *req, ucs_status_t status)
{
    req->status = status;
    ucp_send_request_id_release(req);
    ucs_mpool_put_inline(req->send.rndv.mdesc);

    if (ucp_proto_rndv_request_is_ppln_frag(req)) {
        ucp_proto_rndv_ppln_recv_frag_complete(req, 0, 1);
    } else {
        ucp_proto_rndv_rtr_common_complete(req, UCS_BIT(UCP_DATATYPE_CONTIG));
    }
}

static ucs_status_t ucp_proto_rndv_rtr_mtype_reset(ucp_request_t *req)
{
    ucs_mpool_put_inline(req->send.rndv.mdesc);
    if (ucp_proto_rndv_request_is_ppln_frag(req)) {
        req->status = UCS_ERR_CANCELED;
        ucp_proto_rndv_ppln_recv_frag_clean(req);
        return UCS_ERR_CANCELED;
    }

    return ucp_proto_request_zcopy_id_reset(req);
}

static void ucp_proto_rndv_rtr_mtype_copy_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);
    ucp_proto_rndv_rtr_mtype_complete(req);
}

static void
ucp_proto_rndv_rtr_mtype_data_received(ucp_request_t *req, int in_buffer)
{
    ucp_send_request_id_release(req);
    if (in_buffer) {
        /* Data was already placed in user buffer because the sender responded
           with RNDV_DATA packets */
        ucp_proto_rndv_rtr_mtype_complete(req);
    } else {
        /* Data was not placed in user buffer, which means it was placed to
           the remote address we published - the rendezvous fragment */
        ucp_proto_rndv_mtype_copy(req, req->send.rndv.mdesc->ptr,
                                  ucp_proto_rndv_mtype_get_req_memh(req),
                                  uct_ep_put_zcopy,
                                  ucp_proto_rndv_rtr_mtype_copy_completion,
                                  "out to");
    }
}

static ucs_status_t ucp_proto_rndv_rtr_mtype_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_status_t status;

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        status = ucp_proto_rndv_mtype_request_init(req);
        if (status != UCS_OK) {
            ucp_proto_request_abort(req, status);
            return UCS_OK;
        }

        ucp_proto_rtr_common_request_init(req);
        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    return ucp_proto_rndv_rtr_common_send(req);
}

static ucs_status_t
ucp_proto_rndv_rtr_mtype_init(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_rndv_rtr_priv_t *rpriv = init_params->priv;
    const uint64_t rndv_modes        = UCS_BIT(UCP_RNDV_MODE_PUT_PIPELINE);
    ucp_context_h context            = init_params->worker->context;
    ucs_memory_type_t frag_mem_type  = context->config.ext.rndv_frag_mem_type;
    ucp_proto_perf_node_t *unpack_perf_node;
    ucs_linear_func_t unpack_time;
    ucp_md_map_t md_map, dummy_md_map;
    ucs_status_t status;
    size_t frag_size;
    ucp_md_index_t md_index;

    if (!ucp_proto_rndv_op_check(init_params, UCP_OP_ID_RNDV_RECV, 1) ||
        (init_params->rkey_cfg_index == UCP_WORKER_CFG_INDEX_NULL)) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = ucp_proto_rndv_mtype_init(init_params, &dummy_md_map, &frag_size);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_proto_init_buffer_copy_time(
            init_params->worker, "rtr/mtype unpack", frag_mem_type,
            init_params->select_param->mem_type, UCT_EP_OP_PUT_ZCOPY,
            &unpack_time, &unpack_perf_node);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_mm_get_alloc_md_index(context, &md_index, frag_mem_type);
    if ((status != UCS_OK) || (md_index == UCP_NULL_RESOURCE)) {
        md_map = 0;
    } else {
        md_map = UCS_BIT(md_index);
    }

    status = ucp_proto_rndv_rtr_common_init(init_params, rndv_modes, frag_size,
                                            unpack_time, unpack_perf_node,
                                            md_map, frag_mem_type,
                                            UCS_SYS_DEVICE_ID_UNKNOWN);
    ucp_proto_perf_node_deref(&unpack_perf_node);

    rpriv->pack_cb       = ucp_proto_rndv_rtr_mtype_pack;
    rpriv->data_received = ucp_proto_rndv_rtr_mtype_data_received;

    return status;
}

static void
ucp_proto_rndv_rtr_mtype_query(const ucp_proto_query_params_t *params,
                               ucp_proto_query_attr_t *attr)
{
    const ucp_proto_rndv_ctrl_priv_t *rpriv = params->priv;
    ucp_proto_query_attr_t remote_attr;

    ucp_proto_select_elem_query(params->worker, &rpriv->remote_proto,
                                params->msg_length, &remote_attr);

    attr->is_estimation  = 1;
    attr->max_msg_length = remote_attr.max_msg_length;
    attr->lane_map       = UCS_BIT(rpriv->lane);
    ucp_proto_rndv_mtype_query_desc(params, attr, remote_attr.desc);
    ucs_strncpy_safe(attr->config, remote_attr.config, sizeof(attr->config));
}

ucp_proto_t ucp_rndv_rtr_mtype_proto = {
    .name     = "rndv/rtr/mtype",
    .desc     = NULL,
    .flags    = 0,
    .init     = ucp_proto_rndv_rtr_mtype_init,
    .query    = ucp_proto_rndv_rtr_mtype_query,
    .progress = {ucp_proto_rndv_rtr_mtype_progress},
    .abort    = ucp_proto_rndv_rtr_mtype_abort,
    .reset    = ucp_proto_rndv_rtr_mtype_reset
};

ucs_status_t ucp_proto_rndv_rtr_handle_atp(void *arg, void *data, size_t length,
                                           unsigned flags)
{
    ucp_worker_h worker     = arg;
    ucp_rndv_ack_hdr_t *atp = data;
    const ucp_proto_rndv_rtr_priv_t *rpriv;
    ucp_request_t *req;

    UCP_SEND_REQUEST_GET_BY_ID(&req, worker, atp->super.req_id, 0,
                               return UCS_OK, "ATP %p", atp);

    if (!ucp_proto_common_frag_complete(req, atp->size, "rndv_atp")) {
        return UCS_OK;
    }

    VALGRIND_MAKE_MEM_DEFINED(req->send.state.dt_iter.type.contig.buffer,
                              req->send.state.dt_iter.length);
    rpriv = req->send.proto_config->priv;
    rpriv->data_received(req, 0);
    return UCS_OK;
}

ucs_status_t
ucp_proto_rndv_handle_data(void *arg, void *data, size_t length, unsigned flags)
{
    ucp_worker_h worker                   = arg;
    ucp_request_data_hdr_t *rndv_data_hdr = data;
    size_t recv_len                       = length - sizeof(*rndv_data_hdr);
    const ucp_proto_rndv_rtr_priv_t *rpriv;
    ucs_status_t status;
    ucp_request_t *req;

    UCP_SEND_REQUEST_GET_BY_ID(&req, worker, rndv_data_hdr->req_id, 0,
                               return UCS_OK, "RNDV_DATA %p", rndv_data_hdr);

    status = ucp_datatype_iter_unpack(&req->send.state.dt_iter, worker,
                                      recv_len, rndv_data_hdr->offset,
                                      rndv_data_hdr + 1);
    if (status != UCS_OK) {
        ucp_proto_request_abort(req, status);
        return UCS_OK;
    }

    if (ucp_proto_common_frag_complete(req, recv_len, "rndv_data")) {
        rpriv = req->send.proto_config->priv;
        rpriv->data_received(req, 1);
    }

    return UCS_OK;
}
