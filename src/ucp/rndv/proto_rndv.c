/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "proto_rndv.inl"

#include <ucp/proto/proto_common.inl>


static ucp_md_map_t
ucp_proto_rndv_ctrl_reg_md_map(const ucp_proto_rndv_ctrl_init_params_t *params)
{
    ucp_worker_h worker                      = params->super.super.worker;
    const ucp_ep_config_key_t *ep_config_key = params->super.super.ep_config_key;
    const uct_iface_attr_t *iface_attr;
    const uct_md_attr_t *md_attr;
    ucp_md_index_t md_index;
    ucp_md_map_t reg_md_map;
    ucp_lane_index_t lane;

    /* md_map is all lanes which support get_zcopy on the given mem_type and
     * require remote key
     */
    reg_md_map = 0;
    for (lane = 0; lane < ep_config_key->num_lanes; ++lane) {
        if (ep_config_key->lanes[lane].rsc_index == UCP_NULL_RESOURCE) {
            continue;
        }

        /* Check the lane supports get_zcopy */
        iface_attr = ucp_proto_common_get_iface_attr(&params->super.super,
                                                     lane);
        if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_GET_ZCOPY)) {
            continue;
        }

        /* Check the memory domain requires remote key, and capable of
         * registering the memory type
         */
        md_index = ucp_proto_common_get_md_index(&params->super.super, lane);
        md_attr  = &worker->context->tl_mds[md_index].attr;
        if (!(md_attr->cap.flags & UCT_MD_FLAG_NEED_RKEY) ||
            !(md_attr->cap.reg_mem_types & UCS_BIT(params->mem_info.type))) {
            continue;
        }

        reg_md_map |= UCS_BIT(md_index);
    }

    return reg_md_map;
}

/*
 * Select (guess) the protocol that would be used by the remote peer.
 * We report the rendezvous protocol performance according to the protocol we
 * think the remote peer would select.
 */
static ucs_status_t ucp_proto_rndv_ctrl_select_remote_proto(
        const ucp_proto_rndv_ctrl_init_params_t *params,
        const ucp_proto_select_param_t *remote_select_param,
        ucp_proto_rndv_ctrl_priv_t *rpriv)
{
    ucp_worker_h worker                 = params->super.super.worker;
    ucp_worker_cfg_index_t ep_cfg_index = params->super.super.ep_cfg_index;
    ucp_rkey_config_key_t rkey_config_key;
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_proto_select_elem_t *select_elem;
    ucp_rkey_config_t *rkey_config;
    ucs_status_t status;

    /* Construct remote key for remote protocol lookup according to the local
     * buffer properties (since remote side is expected to access the local
     * buffer)
     */
    rkey_config_key.md_map       = rpriv->md_map;
    rkey_config_key.ep_cfg_index = ep_cfg_index;
    rkey_config_key.mem_type     = params->mem_info.type;
    rkey_config_key.sys_dev      = UCS_SYS_DEVICE_ID_UNKNOWN;

    status = ucp_worker_rkey_config_get(worker, &rkey_config_key, NULL,
                                        &rkey_cfg_index);
    if (status != UCS_OK) {
        return status;
    }

    rkey_config = &worker->rkey_config[rkey_cfg_index];
    select_elem = ucp_proto_select_lookup_slow(worker,
                                               &rkey_config->proto_select,
                                               ep_cfg_index, rkey_cfg_index,
                                               remote_select_param);
    if (select_elem == NULL) {
        ucs_debug("%s: did not find protocol for %s",
                  params->super.super.proto_name,
                  ucp_operation_names[params->remote_op_id]);
        return UCS_ERR_UNSUPPORTED;
    }

    rpriv->remote_proto = *select_elem;
    return UCS_OK;
}

ucs_status_t
ucp_proto_rndv_ctrl_init(const ucp_proto_rndv_ctrl_init_params_t *params)
{
    ucp_context_h context             = params->super.super.worker->context;
    ucp_proto_rndv_ctrl_priv_t *rpriv = params->super.super.priv;
    const ucp_proto_perf_range_t *remote_perf_range;
    ucp_proto_select_param_t remote_select_param;
    ucp_proto_perf_range_t *perf_range;
    const uct_iface_attr_t *iface_attr;
    ucs_linear_func_t send_overheads;
    ucs_memory_info_t mem_info;
    ucp_md_index_t md_index;
    ucp_proto_caps_t *caps;
    ucs_status_t status;
    double rts_latency;

    ucs_assert(params->super.flags & UCP_PROTO_COMMON_INIT_FLAG_RESPONSE);
    ucs_assert(!(params->super.flags & UCP_PROTO_COMMON_INIT_FLAG_MAX_FRAG));

    /* Find lane to send the initial message */
    rpriv->lane = ucp_proto_common_find_am_bcopy_lane(&params->super.super);
    if (rpriv->lane == UCP_NULL_LANE) {
        return UCS_ERR_NO_ELEM;
    }

    /* Construct select parameter for the remote protocol */
    if (params->super.super.rkey_config_key == NULL) {
        /* Remote buffer is unknown, assume same params as local */
        remote_select_param          = *params->super.super.select_param;
        remote_select_param.op_id    = params->remote_op_id;
        remote_select_param.op_flags = 0;
    } else {
        /* If we know the remote buffer parameters, these are actually the local
         * parameters for the remote protocol
         */
        mem_info.type    = params->super.super.rkey_config_key->mem_type;
        mem_info.sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
        ucp_proto_select_param_init(&remote_select_param, params->remote_op_id,
                                    0, UCP_DATATYPE_CONTIG, &mem_info, 1);
    }

    /* Initialize estimated memory registration map */
    rpriv->md_map           = ucp_proto_rndv_ctrl_reg_md_map(params);
    rpriv->packed_rkey_size = ucp_rkey_packed_size(context, rpriv->md_map,
                                                   UCS_SYS_DEVICE_ID_UNKNOWN,
                                                   0);

    /* Guess the protocol the remote side will select */
    status = ucp_proto_rndv_ctrl_select_remote_proto(params,
                                                     &remote_select_param,
                                                     rpriv);
    if (status != UCS_OK) {
        return status;
    }

    /* Set send_overheads to the time to send and receive RTS message */
    iface_attr     = ucp_proto_common_get_iface_attr(&params->super.super,
                                                     rpriv->lane);
    rts_latency    = (iface_attr->overhead * 2) +
                     ucp_tl_iface_latency(context, &iface_attr->latency);
    send_overheads = ucs_linear_func_make(rts_latency, 0.0);

    /* Add registration cost to send_overheads */
    ucs_for_each_bit(md_index, rpriv->md_map) {
        ucs_linear_func_add_inplace(&send_overheads,
                                    context->tl_mds[md_index].attr.reg_cost);
    }

    /* Set rendezvous protocol properties */
    *params->super.super.priv_size         = sizeof(ucp_proto_rndv_ctrl_priv_t);
    params->super.super.caps->cfg_thresh   = params->super.cfg_thresh;
    params->super.super.caps->cfg_priority = params->super.cfg_priority;
    params->super.super.caps->min_length   = params->min_length;
    params->super.super.caps->num_ranges   = 0;

    /* Copy performance ranges from the remote protocol, and add overheads */
    remote_perf_range = rpriv->remote_proto.perf_ranges;
    caps              = params->super.super.caps;
    do {
        perf_range             = &caps->ranges[caps->num_ranges];
        perf_range->max_length = remote_perf_range->max_length;

        /* Add send overheads and apply perf_bias */
        perf_range->perf = ucs_linear_func_compose(
                ucs_linear_func_make(0, 1.0 - params->perf_bias),
                ucs_linear_func_add(remote_perf_range->perf, send_overheads));

        ++caps->num_ranges;
    } while ((remote_perf_range++)->max_length != SIZE_MAX);

    return UCS_OK;
}

void ucp_proto_rndv_ctrl_config_str(size_t min_length, size_t max_length,
                                    const void *priv, ucs_string_buffer_t *strb)
{
    const ucp_proto_rndv_ctrl_priv_t *rpriv = priv;
    const ucp_proto_threshold_elem_t *thresh_elem;
    size_t range_start, range_end;
    const ucp_proto_t *proto;
    ucp_md_index_t md_index;
    char str[64];

    /* Print message lane and memory domains list */
    ucs_string_buffer_appendf(strb, "cln:%d md:", rpriv->lane);
    ucs_for_each_bit(md_index, rpriv->md_map) {
        ucs_string_buffer_appendf(strb, "%d,", md_index);
    }
    ucs_string_buffer_rtrim(strb, ",");
    ucs_string_buffer_appendf(strb, " ");

    /* Print estimated remote protocols for each message size */
    thresh_elem = rpriv->remote_proto.thresholds;
    range_start = 0;
    do {
        range_end = thresh_elem->max_msg_length;

        /* Print only protocols within the range provided by {min,max}_length */
        if ((range_end >= min_length) && (range_start <= max_length)) {
            proto = thresh_elem->proto_config.proto;
            ucs_string_buffer_appendf(strb, "%s(", proto->name);
            proto->config_str(range_start, range_end,
                              thresh_elem->proto_config.priv, strb);
            ucs_string_buffer_appendf(strb, ")");

            if (range_end < max_length) {
                ucs_memunits_to_str(thresh_elem->max_msg_length, str,
                                    sizeof(str));
                ucs_string_buffer_appendf(strb, "<=%s<", str);
            }
        }

        ++thresh_elem;
        range_start = range_end + 1;
    } while (range_end < max_length);

    ucs_string_buffer_rtrim(strb, "<");
}

ucs_status_t ucp_proto_rndv_rts_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_h context                    = init_params->worker->context;
    ucp_proto_rndv_ctrl_init_params_t params = {
        .super.super        = *init_params,
        .super.latency      = 0,
        .super.overhead     = 40e-9,
        .super.cfg_thresh   = context->config.ext.rndv_thresh,
        .super.cfg_priority = 60,
        .super.flags        = UCP_PROTO_COMMON_INIT_FLAG_RESPONSE,
        .remote_op_id       = UCP_OP_ID_RNDV_RECV,
        .perf_bias          = context->config.ext.rndv_perf_diff / 100.0,
        .mem_info.type      = init_params->select_param->mem_type,
        .mem_info.sys_dev   = init_params->select_param->sys_dev,
        .min_length         = 0
    };

    UCP_RMA_PROTO_INIT_CHECK(init_params, UCP_OP_ID_TAG_SEND);

    return ucp_proto_rndv_ctrl_init(&params);
}

ucs_status_t ucp_proto_rndv_ack_init(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_rndv_ack_priv_t *apriv = init_params->priv;

    apriv->lane = ucp_proto_common_find_am_bcopy_lane(init_params);
    if (apriv->lane == UCP_NULL_LANE) {
        return UCS_ERR_NO_ELEM;
    }

    return UCS_OK;
}

ucs_linear_func_t
ucp_proto_rndv_ack_time(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context           = init_params->worker->context;
    ucp_proto_rndv_ack_priv_t *apriv = init_params->priv;
    const uct_iface_attr_t *iface_attr;
    double ack_time;

    iface_attr = ucp_proto_common_get_iface_attr(init_params, apriv->lane);
    ack_time   = (iface_attr->overhead * 2) +
                 ucp_tl_iface_latency(context, &iface_attr->latency);

    return ucs_linear_func_make(ack_time, 0);
}

void ucp_proto_rndv_ack_config_str(size_t min_length, size_t max_length,
                                   const void *priv, ucs_string_buffer_t *strb)
{
    const ucp_proto_rndv_ack_priv_t *apriv = priv;

    ucs_string_buffer_appendf(strb, "aln:%d", apriv->lane);
}

ucs_status_t
ucp_proto_rndv_bulk_init(const ucp_proto_multi_init_params_t *init_params)
{
    ucp_proto_rndv_bulk_priv_t *rpriv    = init_params->super.super.priv;
    ucp_proto_multi_init_params_t params = *init_params;
    ucs_status_t status;
    size_t mpriv_size;

    /* Change priv pointer, since proto_multi priv is not the first element in
     * ucp_proto_rndv_bulk_priv_t struct. Later on, we also update priv size.
     */
    params.super.super.priv      = &rpriv->mpriv;
    params.super.super.priv_size = &mpriv_size;

    status = ucp_proto_multi_init(&params);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_proto_rndv_ack_init(&init_params->super.super);
    if (status != UCS_OK) {
        return status;
    }

    /* Update private data size based of ucp_proto_multi_priv_t variable size */
    *init_params->super.super.priv_size =
            ucs_offsetof(ucp_proto_rndv_bulk_priv_t, mpriv) + mpriv_size;
    return UCS_OK;
}

size_t ucp_proto_rndv_pack_ack(void *dest, void *arg)
{
    ucp_request_t *req       = arg;
    ucp_reply_hdr_t *ack_hdr = dest;

    ack_hdr->req_id = req->send.rndv.remote_req_id;
    ack_hdr->status = UCS_OK;

    return sizeof(*ack_hdr);
}

void ucp_proto_rndv_bulk_config_str(size_t min_length, size_t max_length,
                                    const void *priv, ucs_string_buffer_t *strb)
{
    const ucp_proto_rndv_bulk_priv_t *rpriv = priv;

    ucp_proto_multi_config_str(min_length, max_length, &rpriv->mpriv, strb);
    ucs_string_buffer_appendf(strb, " ");
    ucp_proto_rndv_ack_config_str(min_length, max_length, &rpriv->super, strb);
}

static ucs_status_t
ucp_proto_rndv_send_reply(ucp_worker_h worker, ucp_request_t *req,
                          ucp_operation_id_t op_id, uint8_t sg_count,
                          size_t length, const void *rkey_buffer,
                          size_t rkey_length)
{
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_proto_select_param_t sel_param;
    ucp_proto_select_t *proto_select;
    ucs_status_t status;
    ucp_rkey_h rkey;

    ucs_assert((op_id == UCP_OP_ID_RNDV_RECV) ||
               (op_id == UCP_OP_ID_RNDV_SEND));
    ucs_assert(sg_count == 1);

    if (rkey_length > 0) {
        ucs_assert(rkey_buffer != NULL);
        status = ucp_ep_rkey_unpack_internal(req->send.ep, rkey_buffer,
                                             rkey_length, &rkey);
        if (status != UCS_OK) {
            goto err;
        }

        proto_select   = &ucp_rkey_config(worker, rkey)->proto_select;
        rkey_cfg_index = rkey->cfg_index;
    } else {
        /* No remote key, use endpoint protocols */
        proto_select   = &ucp_ep_config(req->send.ep)->proto_select;
        rkey_cfg_index = UCP_WORKER_CFG_INDEX_NULL;
        rkey           = NULL;
    }

    ucp_proto_select_param_init(&sel_param, op_id, 0,
                                req->send.state.dt_iter.dt_class,
                                &req->send.state.dt_iter.mem_info, sg_count);

    status = ucp_proto_request_set_proto(worker, req->send.ep, req,
                                         proto_select, rkey_cfg_index,
                                         &sel_param, length);
    if (status != UCS_OK) {
        goto err_destroy_rkey;
    }

    req->send.rndv.rkey = rkey;

    ucp_trace_req(req,
                  "%s rva 0x%" PRIx64 " rreq_id 0x%" PRIx64 " with protocol %s",
                  ucp_operation_names[op_id], req->send.rndv.remote_address,
                  req->send.rndv.remote_req_id,
                  req->send.proto_config->proto->name);

    ucp_request_send(req, 0);
    return UCS_OK;

err_destroy_rkey:
    if (rkey != NULL) {
        ucp_rkey_destroy(rkey);
    }
err:
    return status;
}

static UCS_F_ALWAYS_INLINE ucp_request_t *
ucp_request_get_super_req(void *request, void *user_data)
{
    ucp_request_t UCS_V_UNUSED *req = (ucp_request_t*)request - 1;
    ucp_request_t *super_req        = user_data;

    ucs_assert(ucp_request_get_super(req) == super_req);
    return super_req;
}

static void ucp_proto_rndv_recv_completion(void *request, ucs_status_t status,
                                           void *user_data)
{
    ucp_request_t *recv_req = ucp_request_get_super_req(request, user_data);

    ucp_request_complete_tag_recv(recv_req, status);
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_check_rkey_length(uint64_t address, size_t rkey_length,
                                 const char *title)
{
    ucs_assertv((ssize_t)rkey_length >= 0, "%s rkey_length=%zd", title,
                (ssize_t)rkey_length);
    ucs_assertv((address != 0) == (rkey_length > 0),
                "%s rts->address=0x%" PRIx64 " rkey_length=%zu", title, address,
                rkey_length);
}

void ucp_proto_rndv_receive(ucp_worker_h worker, ucp_request_t *recv_req,
                            const ucp_rndv_rts_hdr_t *rts,
                            const void *rkey_buffer, size_t rkey_length)
{
    ucs_status_t status;
    ucp_request_t *req;
    uint8_t sg_count;
    size_t length;
    ucp_ep_h ep;

    UCP_WORKER_GET_VALID_EP_BY_ID(&ep, worker, rts->sreq.ep_id, return,
                                  "RTS on non-existing endpoint");

    if (!UCP_DT_IS_CONTIG(recv_req->recv.datatype)) {
        ucs_fatal("non-contiguous types are not supported with rndv protocol");
    }

    req = ucp_request_get(worker);
    if (req == NULL) {
        ucs_error("failed to allocate rendezvous reply");
        return;
    }

    /* Initialize send request */
    req->send.ep                  = ep;
    req->send.rndv.remote_address = rts->address;
    req->send.rndv.remote_req_id  = rts->sreq.req_id;

    if (ucs_likely(rts->size <= recv_req->recv.length)) {
        req->flags   = UCP_REQUEST_FLAG_CALLBACK | UCP_REQUEST_FLAG_RELEASED;
        req->send.cb = ucp_proto_rndv_recv_completion;
        length       = rts->size;
        ucp_proto_rndv_check_rkey_length(rts->address, rkey_length, "rts");
        ucp_request_set_super(req, recv_req);
    } else {
        /* Short receive: complete with error, and send reply to sender */
        ucp_request_complete_tag_recv(recv_req, UCS_ERR_MESSAGE_TRUNCATED);
        req->flags  = UCP_REQUEST_FLAG_RELEASED;
        length      = 0;
        rkey_length = 0; /* Override rkey length to disable data fetch */
    }

    ucp_datatype_iter_init(worker->context, recv_req->recv.buffer, length,
                           recv_req->recv.datatype, length,
                           &req->send.state.dt_iter, &sg_count);

    status = ucp_proto_rndv_send_reply(worker, req, UCP_OP_ID_RNDV_RECV,
                                       sg_count, length, rkey_buffer,
                                       rkey_length);
    if (status != UCS_OK) {
        ucp_datatype_iter_cleanup(&req->send.state.dt_iter, UINT_MAX);
        ucs_mpool_put(req);
        return;
    }
}

static ucs_status_t
ucp_proto_rndv_send_start(ucp_worker_h worker, ucp_request_t *req,
                          const ucp_rndv_rtr_hdr_t *rtr, size_t header_length)
{
    size_t rkey_length = header_length - sizeof(*rtr);
    ucs_status_t status;

    ucp_proto_rndv_check_rkey_length(rtr->address, rkey_length, "rtr");
    req->send.rndv.remote_address = rtr->address;
    req->send.rndv.remote_req_id  = rtr->rreq_id;

    status = ucp_proto_rndv_send_reply(worker, req, UCP_OP_ID_RNDV_SEND, 1,
                                       rtr->size, rtr + 1, rkey_length);
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}

ucs_status_t
ucp_proto_rndv_handle_rtr(void *arg, void *data, size_t length, unsigned flags)
{
    ucp_worker_h worker           = arg;
    const ucp_rndv_rtr_hdr_t *rtr = data;
    ucs_status_t status;
    ucp_request_t *req;

    UCP_SEND_REQUEST_GET_BY_ID(&req, worker, rtr->sreq_id, 1, return UCS_OK,
                               "RTR %p", rtr);

    if (rtr->address == 0) {
        ucs_fatal("RTR without remote address is currently unsupported");
    }

    /* RTR covers the whole send request - use the send request directly */
    ucs_assert(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED);
    ucs_assert(rtr->size == req->send.state.dt_iter.length);
    ucs_assert(rtr->offset == 0);

    req->flags &= ~UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    status      = ucp_proto_rndv_send_start(worker, req, rtr, length);
    if (status != UCS_OK) {
        goto err_request_fail;
    }

    return UCS_OK;

err_request_fail:
    ucp_proto_request_abort(req, status);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_rtr_uct_comp_from_id(ucp_worker_h worker, uint64_t id,
                                    int extract, uct_completion_t **uct_comp_p)
{
    ucs_status_t status;
    void *ptr;

    status = ucs_ptr_map_get(&worker->ptr_map, id, extract, &ptr);
    if (ucs_unlikely((status != UCS_OK) && (status != UCS_ERR_NO_PROGRESS))) {
        return status;
    }

    *uct_comp_p = ptr;
    return UCS_OK;
}

ucs_status_t
ucp_proto_rndv_handle_data(void *arg, void *data, size_t length, unsigned flags)
{
    ucp_worker_h worker                = arg;
    ucp_rndv_data_hdr_t *rndv_data_hdr = data;
    size_t recv_len                    = length - sizeof(*rndv_data_hdr);
    ucp_request_t *req, *recv_req;
    uct_completion_t *uct_comp;
    ucs_status_t status;

    status = ucp_proto_rndv_rtr_uct_comp_from_id(worker, rndv_data_hdr->rreq_id,
                                                 0, &uct_comp);
    if (ucs_unlikely(status != UCS_OK)) {
        ucs_trace_data("worker %p: completion id 0x%" PRIx64
                       " was not found, drop RNDV data %p",
                       worker, rndv_data_hdr->rreq_id, rndv_data_hdr);
        return UCS_OK;
    }

    req      = ucs_container_of(uct_comp, ucp_request_t, send.state.uct_comp);
    recv_req = ucp_request_get_super(req);
    UCS_PROFILE_REQUEST_EVENT(recv_req, "rndv_data_recv", recv_len);

    ucs_assertv(recv_req->recv.remaining >= recv_len,
                "req->recv.remaining=%zu recv_len=%zu",
                recv_req->recv.remaining, recv_len);
    recv_req->recv.remaining -= recv_len;

    /* process data only if the request is not in error state */
    if (ucs_likely(recv_req->status == UCS_OK)) {
        recv_req->status = ucp_request_recv_data_unpack(
                recv_req, rndv_data_hdr + 1, recv_len, rndv_data_hdr->offset,
                recv_req->recv.remaining == 0);
    }

    if (recv_req->recv.remaining == 0) {
        status = ucs_ptr_map_del(&worker->ptr_map, rndv_data_hdr->rreq_id);
        ucs_assert((status == UCS_OK) || (status == UCS_ERR_NO_PROGRESS));

        ucp_proto_rndv_rtr_common_complete(req, recv_req->status);
    }

    return UCS_OK;
}
