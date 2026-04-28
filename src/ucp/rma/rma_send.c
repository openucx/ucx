/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rma.h"
#include "rma.inl"

#include <ucp/dt/dt_contig.h>
#include <ucs/profile/profile.h>
#include <ucs/sys/stubs.h>

#include <ucp/core/ucp_rkey.inl>
#include <ucp/proto/proto_common.inl>


#define UCP_RMA_CHECK_BUFFER(_buffer, _action) \
    do { \
        if (ENABLE_PARAMS_CHECK && ucs_unlikely((_buffer) == NULL)) { \
            _action; \
        } \
    } while (0)


#define UCP_RMA_CHECK_ZERO_LENGTH(_length, _action) \
    do { \
        if ((_length) == 0) { \
            _action; \
        } \
    } while (0)


#define UCP_RMA_CHECK_PTR(_context, _buffer, _length) \
    do { \
        UCP_CONTEXT_CHECK_FEATURE_FLAGS(_context, UCP_FEATURE_RMA, \
                                        return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM)); \
        UCP_RMA_CHECK_ZERO_LENGTH(_length, return NULL); \
        UCP_RMA_CHECK_BUFFER(_buffer, \
                             return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM)); \
    } while (0)


ucs_status_t ucp_put_nbi(ucp_ep_h ep, const void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey)
{
    ucs_status_ptr_t status_ptr;

    status_ptr = ucp_put_nbx(ep, buffer, length, remote_addr, rkey,
                             &ucp_request_null_param);
    if (UCS_PTR_IS_PTR(status_ptr)) {
        ucp_request_free(status_ptr);
        return UCS_INPROGRESS;
    }

    /* coverity[overflow] */
    return UCS_PTR_STATUS(status_ptr);
}

ucs_status_ptr_t ucp_put_nb(ucp_ep_h ep, const void *buffer, size_t length,
                            uint64_t remote_addr, ucp_rkey_h rkey,
                            ucp_send_callback_t cb)
{
    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK,
        .cb.send      = (ucp_send_nbx_callback_t)cb
    };

    return ucp_put_nbx(ep, buffer, length, remote_addr, rkey, &param);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_put_send_short(ucp_ep_h ep, const void *buffer, size_t length,
                   uint64_t remote_addr, ucp_rkey_h rkey,
                   const ucp_request_param_t *param)
{
    const ucp_rkey_config_t *rkey_config;
    uct_rkey_t tl_rkey;
    ucs_status_t status;

    if (ucs_unlikely(param->op_attr_mask & (UCP_OP_ATTR_FIELD_DATATYPE |
                                            UCP_OP_ATTR_FLAG_NO_IMM_CMPL))) {
        return UCS_ERR_NO_RESOURCE;
    }

    rkey_config = ucp_rkey_config(ep->worker, rkey);
    if (ucs_unlikely(!ucp_proto_select_is_short(ep, &rkey_config->put_short,
                                                length))) {
        return UCS_ERR_NO_RESOURCE;
    }

    tl_rkey = ucp_rkey_get_tl_rkey(rkey, rkey_config->put_short.rkey_index);

    if (ucs_unlikely(ucp_ep_rma_is_fence_required(ep))) {
        /* TODO: check support for fence in fast path short */
        return UCS_ERR_NO_RESOURCE;
    }

    status = UCS_PROFILE_CALL(uct_ep_put_short,
                              ucp_ep_get_fast_lane(ep,
                                                   rkey_config->put_short.lane),
                              buffer, length, remote_addr, tl_rkey);
    if (status == UCS_OK) {
        ep->ext->unflushed_lanes |= UCS_BIT(rkey_config->put_short.lane);
    }

    return status;
}

ucs_status_ptr_t ucp_put_nbx(ucp_ep_h ep, const void *buffer, size_t count,
                             uint64_t remote_addr, ucp_rkey_h rkey,
                             const ucp_request_param_t *param)
{
    ucp_worker_h worker     = ep->worker;
    size_t contig_length    = 0;
    ucp_datatype_t datatype = ucp_dt_make_contig(1);
    ucs_status_ptr_t ret;
    ucs_status_t status;
    ucp_request_t *req;

    UCP_REQUEST_CHECK_PARAM(param);
    UCP_RMA_CHECK_PTR(worker->context, buffer, count);
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    ucs_trace_req("put_nbx buffer %p count %zu remote_addr %" PRIx64
                  " rkey %p to %s cb %p",
                  buffer, count, remote_addr, rkey, ucp_ep_peer_name(ep),
                  ucp_request_param_send_callback(param));

    if (ucs_unlikely(!worker->context->config.ext.proto_enable)) {
        ret = UCS_STATUS_PTR(UCS_ERR_UNSUPPORTED);
        goto out_unlock;
    }

    status = ucp_put_send_short(ep, buffer, count, remote_addr, rkey, param);
    if (ucs_likely(status != UCS_ERR_NO_RESOURCE) ||
        ucs_unlikely(param->op_attr_mask & UCP_OP_ATTR_FLAG_FORCE_IMM_CMPL)) {
        ret = UCS_STATUS_PTR(status);
        goto out_unlock;
    }

    req = ucp_request_get_param(worker, param,
                                {ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
                                goto out_unlock;});
    req->send.rma.rkey        = rkey;
    req->send.rma.remote_addr = remote_addr;

    if (ucs_unlikely(param->op_attr_mask & UCP_OP_ATTR_FIELD_DATATYPE)) {
        datatype = param->datatype;
        if (UCP_DT_IS_CONTIG(datatype)) {
            contig_length = ucp_contig_dt_length(datatype, count);
        }
    } else {
        contig_length = count;
    }

    ret = ucp_proto_request_send_op_rma(
            ep, rkey, req, ucp_ep_rma_get_fence_flag(ep), UCP_OP_ID_PUT,
            buffer, count, datatype, contig_length, param, 0, 0);

out_unlock:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    return ret;
}

ucs_status_t ucp_get_nbi(ucp_ep_h ep, void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey)
{
    ucs_status_ptr_t status_ptr;

    status_ptr = ucp_get_nbx(ep, buffer, length, remote_addr, rkey,
                             &ucp_request_null_param);
    if (UCS_PTR_IS_PTR(status_ptr)) {
        ucp_request_free(status_ptr);
        return UCS_INPROGRESS;
    }

    /* coverity[overflow] */
    return UCS_PTR_STATUS(status_ptr);
}

ucs_status_ptr_t ucp_get_nb(ucp_ep_h ep, void *buffer, size_t length,
                            uint64_t remote_addr, ucp_rkey_h rkey,
                            ucp_send_callback_t cb)
{
    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK,
        .cb.send      = (ucp_send_nbx_callback_t)cb
    };

    return ucp_get_nbx(ep, buffer, length, remote_addr, rkey, &param);
}

ucs_status_ptr_t ucp_get_nbx(ucp_ep_h ep, void *buffer, size_t count,
                             uint64_t remote_addr, ucp_rkey_h rkey,
                             const ucp_request_param_t *param)
{
    ucp_worker_h worker  = ep->worker;
    size_t contig_length = 0;
    ucs_status_ptr_t ret;
    ucp_request_t *req;
    uintptr_t datatype;

    if (ucs_unlikely(param->op_attr_mask & UCP_OP_ATTR_FLAG_FORCE_IMM_CMPL)) {
        return UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE);
    }

    UCP_REQUEST_CHECK_PARAM(param);
    UCP_RMA_CHECK_PTR(worker->context, buffer, count);
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    ucs_trace_req("get_nbx buffer %p count %zu remote_addr %" PRIx64
                  " rkey %p from %s cb %p",
                  buffer, count, remote_addr, rkey, ucp_ep_peer_name(ep),
                  ucp_request_param_send_callback(param));

    if (ucs_unlikely(!worker->context->config.ext.proto_enable)) {
        ret = UCS_STATUS_PTR(UCS_ERR_UNSUPPORTED);
        goto out_unlock;
    }

    datatype = ucp_request_param_datatype(param);
    req      = ucp_request_get_param(worker, param,
                                     {ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
                                     goto out_unlock;});

    req->send.rma.rkey             = rkey;
    req->send.rma.remote_addr      = remote_addr;
    req->send.state.completed_size = 0;
    if (UCP_DT_IS_CONTIG(datatype)) {
        contig_length = ucp_contig_dt_length(datatype, count);
    }

    ret = ucp_proto_request_send_op_rma(
            ep, rkey, req, ucp_ep_rma_get_fence_flag(ep), UCP_OP_ID_GET,
            buffer, count, datatype, contig_length, param, 0, 0);

out_unlock:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    return ret;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_put, (ep, buffer, length, remote_addr, rkey),
                 ucp_ep_h ep, const void *buffer, size_t length,
                 uint64_t remote_addr, ucp_rkey_h rkey)
{
    return ucp_rma_wait(ep->worker,
                        ucp_put_nb(ep, buffer, length, remote_addr, rkey,
                                   (ucp_send_callback_t)ucs_empty_function),
                        "put");
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_get, (ep, buffer, length, remote_addr, rkey),
                 ucp_ep_h ep, void *buffer, size_t length,
                 uint64_t remote_addr, ucp_rkey_h rkey)
{
    return ucp_rma_wait(ep->worker,
                        ucp_get_nb(ep, buffer, length, remote_addr, rkey,
                                   (ucp_send_callback_t)ucs_empty_function),
                        "get");
}
