/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "rocm_copy_ep.h"
#include "rocm_copy_iface.h"

#include <uct/base/uct_log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>
#include <ucs/arch/cpu.h>

#include <hip/hip_runtime_api.h>

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rocm_memcpy_h2d(uct_ep_h tl_ep, void *dst, const void *src, size_t len)
{
    uct_rocm_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rocm_copy_iface_t);

    if (ucs_likely(len < iface->config.h2d_thresh)) {
        memcpy(dst, src, len);
    } else {
        if (hipSuccess != hipMemcpy(dst, src, len, hipMemcpyHostToDevice)) {
            ucs_error("failed to copy %ld bytes from %p to %p", len, src, dst);
            return UCS_ERR_IO_ERROR;
        }
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rocm_memcpy_d2h(uct_ep_h tl_ep, void *dst, const void *src, size_t len)
{
    uct_rocm_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rocm_copy_iface_t);

    if (ucs_likely(len < iface->config.d2h_thresh)) {
        ucs_memcpy_nontemporal(dst, src, len);
    } else {
        if (hipSuccess != hipMemcpy(dst, src, len, hipMemcpyDeviceToHost)) {
            ucs_error("failed to copy %ld bytes from %p to %p", len, src, dst);
            return UCS_ERR_IO_ERROR;
        }
    }

    return UCS_OK;
}

static UCS_CLASS_INIT_FUNC(uct_rocm_copy_ep_t, const uct_ep_params_t *params)
{
    uct_rocm_copy_iface_t *iface = ucs_derived_of(params->iface, uct_rocm_copy_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rocm_copy_ep_t)
{
}

UCS_CLASS_DEFINE(uct_rocm_copy_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_rocm_copy_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rocm_copy_ep_t, uct_ep_t);

#define uct_rocm_copy_trace_data(_remote_addr, _rkey, _fmt, ...) \
     ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_remote_addr), \
                    (_rkey))

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rocm_copy_ep_zcopy(uct_ep_h tl_ep,
                                   uint64_t remote_addr,
                                   const uct_iov_t *iov,
                                   int is_put)
{
    size_t size = uct_iov_get_length(iov);

    if (!size) {
        return UCS_OK;
    }

    if (is_put) {
        return uct_rocm_memcpy_h2d(tl_ep, (void *)remote_addr, iov->buffer, size);
    } else {
        return uct_rocm_memcpy_d2h(tl_ep, iov->buffer, (void *)remote_addr, size);
    }
}

ucs_status_t uct_rocm_copy_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                        uint64_t remote_addr, uct_rkey_t rkey,
                                        uct_completion_t *comp)
{
    ucs_status_t status;

    status = uct_rocm_copy_ep_zcopy(tl_ep, remote_addr, iov, 0);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_rocm_copy_trace_data(remote_addr, rkey, "GET_ZCOPY [length %zu]",
                             uct_iov_total_length(iov, iovcnt));
    return status;
}

ucs_status_t uct_rocm_copy_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                        uint64_t remote_addr, uct_rkey_t rkey,
                                        uct_completion_t *comp)
{
    ucs_status_t status;

    status = uct_rocm_copy_ep_zcopy(tl_ep, remote_addr, iov, 1);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_rocm_copy_trace_data(remote_addr, rkey, "GET_ZCOPY [length %zu]",
                             uct_iov_total_length(iov, iovcnt));
    return status;
}


ucs_status_t uct_rocm_copy_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                        unsigned length, uint64_t remote_addr,
                                        uct_rkey_t rkey)
{
    ucs_status_t status;

    status = uct_rocm_memcpy_h2d(tl_ep, (void *)remote_addr, buffer, length);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, SHORT, length);
    ucs_trace_data("PUT_SHORT size %d from %p to %p",
                   length, buffer, (void *)remote_addr);
    return status;
}

ucs_status_t uct_rocm_copy_ep_get_short(uct_ep_h tl_ep, void *buffer,
                                        unsigned length, uint64_t remote_addr,
                                        uct_rkey_t rkey)
{
    ucs_status_t status;

    status = uct_rocm_memcpy_d2h(tl_ep, buffer, (void *)remote_addr, length);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, SHORT, length);
    ucs_trace_data("GET_SHORT size %d from %p to %p",
                   length, (void *)remote_addr, buffer);
    return status;
}
