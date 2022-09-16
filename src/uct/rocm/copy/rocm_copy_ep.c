/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019-2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rocm_copy_ep.h"
#include "rocm_copy_iface.h"
#include "rocm_copy_md.h"
#include "rocm_copy_cache.h"


#include <uct/rocm/base/rocm_base.h>
#include <uct/base/uct_log.h>
#include <uct/base/uct_iov.inl>
#include <ucs/debug/memtrack_int.h>
#include <ucs/type/class.h>
#include <ucs/arch/cpu.h>

#include <hsa_ext_amd.h>

#define uct_rocm_memcpy_h2d(_d,_s,_l)  memcpy((_d),(_s),(_l))
#define uct_rocm_memcpy_d2h(_d,_s,_l)  ucs_memcpy_nontemporal((_d),(_s),(_l))

static UCS_CLASS_INIT_FUNC(uct_rocm_copy_ep_t, const uct_ep_params_t *params)
{
    uct_rocm_copy_iface_t *iface = ucs_derived_of(params->iface, uct_rocm_copy_iface_t);
    char target_name[64];
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    snprintf(target_name, sizeof(target_name), "dest:%d",
             *(pid_t*)params->iface_addr);
    status = uct_rocm_copy_create_cache(&self->local_memh_cache, target_name);
    if (status != UCS_OK) {
        ucs_error("could not create create rocm copy cache: %s",
                  ucs_status_string(status));
    }

    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rocm_copy_ep_t)
{
    uct_rocm_copy_destroy_cache(self->local_memh_cache);
}

UCS_CLASS_DEFINE(uct_rocm_copy_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_rocm_copy_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rocm_copy_ep_t, uct_ep_t);

#define uct_rocm_copy_trace_data(_remote_addr, _rkey, _fmt, ...) \
     ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_remote_addr), \
                    (_rkey))

static void*
uct_rocm_copy_get_mapped_host_ptr (uct_ep_h tl_ep, void *ptr, size_t size,
                                   uct_rocm_copy_key_t *rocm_copy_key)
{
    uct_rocm_copy_ep_t *ep = ucs_derived_of(tl_ep, uct_rocm_copy_ep_t);
    size_t offset          = 0;

    void *mapped_ptr;
    ucs_status_t status;

    if ((void*)rocm_copy_key->vaddr == rocm_copy_key->dev_ptr) {
        /* key contains rocm address information. Need to lock host address first */
        status = uct_rocm_copy_cache_map_memhandle(ep->local_memh_cache,
                                                   (uint64_t)ptr,
                                                   size, &mapped_ptr);
        if ((status != UCS_OK) || (mapped_ptr == NULL)) {
            ucs_trace("Failed to lock memory addr %p", ptr);
            return NULL;
        }
    }
    else {
        /* rkey contains host address information */
        offset     = (uint64_t) ptr - rocm_copy_key->vaddr;
        mapped_ptr = UCS_PTR_BYTE_OFFSET(rocm_copy_key->dev_ptr, offset);
    }

    return mapped_ptr;
}

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
    void *remote_addr_mod=NULL, *iov_buffer_mod=NULL;
    ucs_status_t stat;
    ucs_memory_type_t mem_type;

    ucs_trace("remote addr %p rkey %p size %zu",
              (void*)remote_addr, (void*)rkey, size);

    stat = uct_rocm_base_detect_memory_type((uct_md_h)0, (const void *)remote_addr,
                                            size, &mem_type);
    if (stat != UCS_OK) {
        return UCS_ERR_IO_ERROR;
    }
    if (mem_type == UCS_MEMORY_TYPE_ROCM) {
        remote_addr_mod = (void*)remote_addr;

        /* need agent */
        uct_rocm_base_get_ptr_info(remote_addr_mod,  size, NULL, NULL, &agent);
    } else {
        remote_addr_mod = uct_rocm_copy_get_mapped_host_ptr(tl_ep, (void*)remote_addr,
                                                            size, rocm_copy_key);
        if (remote_addr_mod == NULL) {
            ucs_error("failed to map host pointer to device address");
            return UCS_ERR_IO_ERROR;
        }
    }

    stat = uct_rocm_base_detect_memory_type((uct_md_h)0, (const void *)iov->buffer,
                                            size, &mem_type);
    if (stat != UCS_OK) {
        return UCS_ERR_IO_ERROR;
    }
    if (mem_type == UCS_MEMORY_TYPE_ROCM) {
        iov_buffer_mod  = (void*)iov->buffer;

        /* need agent */
        uct_rocm_base_get_ptr_info(iov_buffer_mod,  size, NULL, NULL, &agent);
    } else {
        iov_buffer_mod = uct_rocm_copy_get_mapped_host_ptr(tl_ep, (void*)iov->buffer,
                                                           size, rocm_copy_key);
        if (iov_buffer_mod == NULL) {
            ucs_error("failed to map host ptr to device address");
            return UCS_ERR_IO_ERROR;
        }
    }

    if ( (iov_buffer_mod != iov->buffer) && (remote_addr_mod != (void*)remote_addr)) {
        /*
         * both pointers are host addresses. This can happen in send-to-self scenarios.
         * for a rocm addres, the modified buffer pointer is identical to the original
         * value.
         * using the original host addresses, not the mapped addresses.
         */
        if (is_put) {
            src_addr = iov->buffer;
            dst_addr = (void*)remote_addr;
        } else {
            src_addr = (void*)remote_addr;
            dst_addr = iov->buffer;
        }
        ucs_trace("Executing memcpy from %p to %p size %zu\n", src_addr, dst_addr, size);
        memcpy(dst_addr, src_addr, size);

        return UCS_OK;
    }

    if (is_put) {
        src_addr = iov_buffer_mod;
        dst_addr = remote_addr_mod;
    } else {
        src_addr = remote_addr_mod;
        dst_addr = iov_buffer_mod;
    }

    hsa_signal_store_screlease(signal, 1);
    status = hsa_amd_memory_async_copy(dst_addr, agent,
                                       src_addr, agent,
                                       size, 0, NULL, signal);

    while (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                     UINT64_MAX, HSA_WAIT_STATE_ACTIVE));
    ucs_trace("hsa async copy from src %p to dst %p, len %ld status %d",
              src_addr, dst_addr, size, (int)status);
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
    uct_rocm_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rocm_copy_iface_t);
    ucs_status_t status          = UCS_OK;
    uct_iov_t *iov;

    if (length <= iface->config.short_h2d_thresh) {
        uct_rocm_memcpy_h2d((void*)remote_addr, buffer, length);
    } else {
        iov = ucs_malloc(sizeof(uct_iov_t), "uct_iov_t");
        if (iov == NULL) {
            ucs_error("failed to allocate memory for uct_iov_t");
            return UCS_ERR_NO_MEMORY;
        }

        iov->buffer = (void*)buffer;
        iov->length = length;
        iov->count  = 1;
        status      = uct_rocm_copy_ep_zcopy(tl_ep, remote_addr, iov, rkey, 1);
        if (status != UCS_OK) {
            ucs_error("error in uct_rocm_copy_ep_zcopy %s",
                      ucs_status_string(status));
        }

        ucs_free(iov);
    }

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, SHORT, length);
    ucs_trace_data("PUT_SHORT size %d from %p to %p",
                   length, buffer, (void *)remote_addr);
    return status;
}

ucs_status_t uct_rocm_copy_ep_get_short(uct_ep_h tl_ep, void *buffer,
                                        unsigned length, uint64_t remote_addr,
                                        uct_rkey_t rkey)
{
    uct_rocm_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rocm_copy_iface_t);
    ucs_status_t status          = UCS_OK;
    uct_iov_t *iov;

    if (length <= iface->config.short_d2h_thresh) {
        uct_rocm_memcpy_d2h(buffer, (void*)remote_addr, length);
    } else {
        iov = ucs_malloc(sizeof(uct_iov_t), "uct_iov_t");
        if (iov == NULL) {
            ucs_error("failed to allocate memory for uct_iov_t");
            return UCS_ERR_NO_MEMORY;
        }

        iov->buffer = buffer;
        iov->length = length;
        iov->count  = 1;
        status      = uct_rocm_copy_ep_zcopy(tl_ep, remote_addr, iov, rkey, 0);
        if (status != UCS_OK) {
            ucs_error("error in uct_rocm_copy_ep_zcopy %s",
                      ucs_status_string(status));
        }

        ucs_free(iov);
    }

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, SHORT, length);
    ucs_trace_data("GET_SHORT size %d from %p to %p",
                   length, (void *)remote_addr, buffer);
    return status;
}
