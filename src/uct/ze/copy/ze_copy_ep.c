/*
 * Copyright (C) Intell Corporation, 2023-2024. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ze_copy_ep.h"
#include "ze_copy_iface.h"
#include "ze_copy_md.h"

#include <uct/ze/base/ze_base.h>
#include <uct/base/uct_log.h>
#include <uct/base/uct_iov.inl>
#include <ucs/debug/memtrack_int.h>
#include <ucs/type/class.h>
#include <ucs/arch/cpu.h>

#include <level_zero/ze_api.h>

static UCS_CLASS_INIT_FUNC(uct_ze_copy_ep_t, const uct_ep_params_t *params)
{
    uct_ze_copy_iface_t *iface = ucs_derived_of(params->iface,
                                                uct_ze_copy_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ze_copy_ep_t)
{
}

UCS_CLASS_DEFINE(uct_ze_copy_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_ze_copy_ep_t, uct_ep_t, const uct_ep_params_t*);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_ze_copy_ep_t, uct_ep_t);

ucs_status_t uct_ze_copy_ep_zcopy(uct_ep_h tl_ep, uint64_t remote_addr,
                                  const uct_iov_t *iov, uct_rkey_t rkey,
                                  int is_put)
{
    size_t size                = uct_iov_get_length(iov);
    uct_ze_copy_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                uct_ze_copy_iface_t);
    ze_result_t ret;
    void *src, *dst;

    ucs_trace("remote addr %p rkey %p size %zu", (void*)remote_addr,
              (void*)rkey, size);

    if (is_put) {
        src = iov->buffer;
        dst = (void*)remote_addr;
    } else {
        src = (void*)remote_addr;
        dst = iov->buffer;
    }

    ret = zeCommandListAppendMemoryCopy(iface->ze_cmdl, dst, src, size, NULL, 0,
                                        NULL);
    if (ret != ZE_RESULT_SUCCESS) {
        return UCS_ERR_IO_ERROR;
    }

    ret = zeCommandListClose(iface->ze_cmdl);
    if (ret != ZE_RESULT_SUCCESS) {
        return UCS_ERR_IO_ERROR;
    }

    ret = zeCommandQueueExecuteCommandLists(iface->ze_cmdq, 1, &iface->ze_cmdl,
                                            NULL);
    if (ret != ZE_RESULT_SUCCESS) {
        return UCS_ERR_IO_ERROR;
    }

    ret = zeCommandQueueSynchronize(iface->ze_cmdq, UINT32_MAX);
    if (ret != ZE_RESULT_SUCCESS) {
        return UCS_ERR_IO_ERROR;
    }

    ret = zeCommandListReset(iface->ze_cmdl);
    if (ret != ZE_RESULT_SUCCESS) {
        return UCS_ERR_IO_ERROR;
    }

    ucs_trace("ze memory copy from src %p to dst %p, len %ld", src, dst, size);

    return UCS_OK;
}

ucs_status_t uct_ze_copy_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                      size_t iovcnt, uint64_t remote_addr,
                                      uct_rkey_t rkey, uct_completion_t *comp)
{
    ucs_status_t status;

    status = uct_ze_copy_ep_zcopy(tl_ep, remote_addr, iov, rkey, 0);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    ucs_trace_data("GET_ZCOPY size %zu from %p (%+ld)",
                   uct_iov_total_length(iov, iovcnt), (void *)remote_addr,
                   rkey);
    return status;
}

ucs_status_t uct_ze_copy_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                      size_t iovcnt, uint64_t remote_addr,
                                      uct_rkey_t rkey, uct_completion_t *comp)
{
    ucs_status_t status;

    status = uct_ze_copy_ep_zcopy(tl_ep, remote_addr, iov, rkey, 1);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    ucs_trace_data("PUT_ZCOPY size %zu to %p (%+ld)",
                   uct_iov_total_length(iov, iovcnt), (void *)remote_addr,
                   rkey);
    return status;
}

ucs_status_t uct_ze_copy_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                      unsigned length, uint64_t remote_addr,
                                      uct_rkey_t rkey)
{
    uct_iov_t iov = {
        .buffer = (void*)buffer,
        .length = length,
        .count  = 1,
    };
    ucs_status_t status;

    status = uct_ze_copy_ep_zcopy(tl_ep, remote_addr, &iov, rkey, 1);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, SHORT, length);
    ucs_trace_data("PUT_SHORT size %u from %p to %p", length, buffer,
                   (void*)remote_addr);
    return status;
}

ucs_status_t uct_ze_copy_ep_get_short(uct_ep_h tl_ep, void *buffer,
                                      unsigned length, uint64_t remote_addr,
                                      uct_rkey_t rkey)
{
    uct_iov_t iov = {
        .buffer = buffer,
        .length = length,
        .count  = 1,
    };
    ucs_status_t status;

    status = uct_ze_copy_ep_zcopy(tl_ep, remote_addr, &iov, rkey, 0);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, SHORT, length);
    ucs_trace_data("GET_SHORT size %u from %p to %p", length,
                   (void*)remote_addr, buffer);
    return status;
}
