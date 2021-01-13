/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rocm_copy_ep.h"
#include "rocm_copy_iface.h"
#include "rocm_copy_md.h"


#include <uct/rocm/base/rocm_base.h>
#include <uct/base/uct_log.h>
#include <uct/base/uct_iov.inl>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>
#include <ucs/arch/cpu.h>

#include <hsa_ext_amd.h>

#define uct_rocm_memcpy_h2d(_d,_s,_l)  memcpy((_d),(_s),(_l))
#define uct_rocm_memcpy_d2h(_d,_s,_l)  ucs_memcpy_nontemporal((_d),(_s),(_l))

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

ucs_status_t uct_rocm_copy_ep_zcopy(uct_ep_h tl_ep,
                                    uint64_t remote_addr,
                                    const uct_iov_t *iov,
                                    uct_rkey_t rkey,
                                    int is_put)
{
    size_t size                        = uct_iov_get_length(iov);
    uct_rocm_copy_iface_t *iface       = ucs_derived_of(tl_ep->iface, uct_rocm_copy_iface_t);
    hsa_signal_t signal                = iface->hsa_signal;
    uct_rocm_copy_key_t *rocm_copy_key = (uct_rocm_copy_key_t *) rkey;

    hsa_status_t status;
    hsa_agent_t agent;
    void *src_addr, *dst_addr;
    void *host_ptr, *dev_ptr, *mapped_ptr;
    size_t offset;

    ucs_trace("remote addr %p rkey %p size %zu",
              (void*)remote_addr, (void*)rkey, size);

    if (is_put) {   /* Host-to-Device */
        host_ptr = iov->buffer;
        dev_ptr  = (void *)remote_addr;
    } else {        /* Device-to-Host */
        dev_ptr  = (void *)remote_addr;
        host_ptr = iov->buffer;
    }

    offset     = (uint64_t) host_ptr - rocm_copy_key->vaddr;
    mapped_ptr = UCS_PTR_BYTE_OFFSET(rocm_copy_key->dev_ptr, offset);

    ucs_trace("host_ptr %p offset %zu dev_ptr %p mapped_ptr %p",
              host_ptr, offset, rocm_copy_key->dev_ptr, mapped_ptr);

    status = uct_rocm_base_get_ptr_info(dev_ptr,  size, NULL, NULL, &agent);
    if (status != HSA_STATUS_SUCCESS) {
        const char *addr_type = is_put ? "DST" : "SRC";
        ucs_error("%s addr %p/%lx is not ROCM memory", addr_type, dev_ptr, size);
        return UCS_ERR_INVALID_ADDR;
    }

    if (is_put) {
        src_addr = mapped_ptr;
        dst_addr = dev_ptr;
    } else {
        src_addr = dev_ptr;
        dst_addr = mapped_ptr;
    }

    hsa_signal_store_screlease(signal, 1);
    ucs_trace("hsa async copy from src %p to dst %p, len %ld",
              src_addr, dst_addr, size);
    status = hsa_amd_memory_async_copy(dst_addr, agent,
                                       src_addr, agent,
                                       size, 0, NULL, signal);

    while (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                     UINT64_MAX, HSA_WAIT_STATE_ACTIVE));
    return UCS_OK;
}

ucs_status_t uct_rocm_copy_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                        uint64_t remote_addr, uct_rkey_t rkey,
                                        uct_completion_t *comp)
{
    size_t size                  = uct_iov_get_length(iov);
    uct_rocm_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rocm_copy_iface_t);
    ucs_status_t status;

    if (size < iface->config.d2h_thresh) {
        uct_rocm_memcpy_d2h(iov->buffer, (void *)remote_addr, size);
        status = UCS_OK;
    } else {
        status = uct_rocm_copy_ep_zcopy(tl_ep, remote_addr, iov, rkey, 0);
    }

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
    size_t size                  = uct_iov_get_length(iov);
    uct_rocm_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rocm_copy_iface_t);
    ucs_status_t status;

    if (size < iface->config.h2d_thresh) {
        uct_rocm_memcpy_h2d((void *)remote_addr, iov->buffer, size);
        status = UCS_OK;
    } else {
        status = uct_rocm_copy_ep_zcopy(tl_ep, remote_addr, iov, rkey, 1);
    }

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
    uct_rocm_memcpy_h2d((void *)remote_addr, buffer, length);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, SHORT, length);
    ucs_trace_data("PUT_SHORT size %d from %p to %p",
                   length, buffer, (void *)remote_addr);
    return UCS_OK;
}

ucs_status_t uct_rocm_copy_ep_get_short(uct_ep_h tl_ep, void *buffer,
                                        unsigned length, uint64_t remote_addr,
                                        uct_rkey_t rkey)
{
    uct_rocm_memcpy_d2h(buffer, (void *)remote_addr, length);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, SHORT, length);
    ucs_trace_data("GET_SHORT size %d from %p to %p",
                   length, (void *)remote_addr, buffer);
    return UCS_OK;
}
