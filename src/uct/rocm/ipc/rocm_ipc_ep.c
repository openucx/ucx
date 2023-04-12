/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019-2023. ALL RIGHTS RESERVED.
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rocm_ipc_ep.h"
#include "rocm_ipc_iface.h"
#include "rocm_ipc_md.h"

#include <uct/rocm/base/rocm_base.h>
#include <uct/rocm/base/rocm_signal.h>
#include <uct/base/uct_iov.inl>
#include <ucs/profile/profile.h>

static UCS_CLASS_INIT_FUNC(uct_rocm_ipc_ep_t, const uct_ep_params_t *params)
{
    uct_rocm_ipc_iface_t *iface = ucs_derived_of(params->iface, uct_rocm_ipc_iface_t);
    char target_name[64];
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    self->remote_pid = *(const pid_t*)params->iface_addr;

    snprintf(target_name, sizeof(target_name), "dest:%d", *(pid_t*)params->iface_addr);
    status = uct_rocm_ipc_create_cache(&self->remote_memh_cache, target_name);
    if (status != UCS_OK) {
        ucs_error("could not create create rocm ipc cache: %s",
                  ucs_status_string(status));
        return status;
    }

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rocm_ipc_ep_t)
{
    uct_rocm_ipc_destroy_cache(self->remote_memh_cache);
}

UCS_CLASS_DEFINE(uct_rocm_ipc_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rocm_ipc_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rocm_ipc_ep_t, uct_ep_t);

#define uct_rocm_ipc_trace_data(_remote_addr, _rkey, _fmt, ...) \
    ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_remote_addr), \
                   (_rkey))

ucs_status_t uct_rocm_ipc_ep_zcopy(uct_ep_h tl_ep,
                                   uint64_t remote_addr,
                                   const uct_iov_t *iov,
                                   uct_rocm_ipc_key_t *key,
                                   uct_completion_t *comp,
                                   int is_put)
{
    uct_rocm_ipc_ep_t *ep = ucs_derived_of(tl_ep, uct_rocm_ipc_ep_t);
    hsa_status_t status;
    hsa_agent_t local_agent, remote_agent;
    hsa_agent_t dst_agent, src_agent;
    size_t size = uct_iov_get_length(iov);
    ucs_status_t ret = UCS_OK;
    void *base_addr, *local_addr = iov->buffer;
    uct_rocm_ipc_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rocm_ipc_iface_t);
    void *remote_base_addr, *remote_copy_addr;
    void *dst_addr, *src_addr;
    uct_rocm_base_signal_desc_t *rocm_ipc_signal;
    void *tmp_base_ptr;
    size_t tmp_base_size;
    hsa_agent_t *gpu_agents;
    hsa_amd_pointer_type_t mem_type;
    int num_gpu;

    /* no data to deliver */
    if (!size)
        return UCS_OK;

    if ((remote_addr < key->address) ||
        (remote_addr + size > key->address + key->length)) {
        ucs_error("remote addr %lx/%lx out of range %lx/%lx",
                  remote_addr, size, key->address, key->length);
        return UCS_ERR_INVALID_PARAM;
    }

    status = uct_rocm_base_get_ptr_info(local_addr, size, &base_addr, NULL,
                                        &mem_type, &local_agent, NULL);
    if ((status != HSA_STATUS_SUCCESS) ||
        (mem_type == HSA_EXT_POINTER_TYPE_UNKNOWN)) {
        ucs_error("local addr %p/%lx is not ROCM memory", local_addr, size);
        return UCS_ERR_INVALID_ADDR;
    }

    ret = uct_rocm_ipc_cache_map_memhandle((void *)ep->remote_memh_cache, key,
                                           &remote_base_addr);
    if (ret != UCS_OK) {
        ucs_error("fail to attach ipc mem %p %d\n", (void *)key->address, ret);
        return ret;
    }

    remote_copy_addr = UCS_PTR_BYTE_OFFSET(remote_base_addr,
                                           remote_addr - key->address);

    memset(&remote_agent, 0, sizeof(hsa_agent_t));
    status = uct_rocm_base_get_ptr_info(remote_copy_addr, size, &tmp_base_ptr,
                                        &tmp_base_size, &mem_type,
                                        &remote_agent, NULL);
    if ((status != HSA_STATUS_SUCCESS) ||
        (mem_type == HSA_EXT_POINTER_TYPE_UNKNOWN)) {
        ucs_error("remote addr %p %lu is not ROCM memory status=%d mem_type %d",
                  remote_copy_addr, size, status, mem_type);
        return UCS_ERR_INVALID_ADDR;
    }

    if (remote_agent.handle == 0) {
        /* No access to remote agent, e.g. because of limited visability of devices to
         * this process. Using local_agent as a backup plan. */
        remote_agent = local_agent;
    } else {
        num_gpu = uct_rocm_base_get_gpu_agents(&gpu_agents);
        status  = UCS_PROFILE_CALL_ALWAYS(hsa_amd_agents_allow_access, num_gpu,
                                          gpu_agents, NULL, base_addr);
        if (status != HSA_STATUS_SUCCESS) {
            ucs_error("failed to enable direct access for mem addr %p agent "
                      "%lu\n",
                      (void*)remote_addr, remote_agent.handle);
            return UCS_ERR_INVALID_ADDR;
        }
    }
    if (is_put) {
        dst_addr  = remote_copy_addr;
        dst_agent = remote_agent;
        src_addr  = local_addr;
        src_agent = local_agent;
    } else {
        dst_addr  = local_addr;
        dst_agent = local_agent;
        src_addr  = remote_copy_addr;
        src_agent = remote_agent;
    }

    rocm_ipc_signal = ucs_mpool_get(&iface->signal_pool);
    hsa_signal_store_screlease(rocm_ipc_signal->signal, 1);

    status = UCS_PROFILE_CALL_ALWAYS(hsa_amd_memory_async_copy, dst_addr,
                                     dst_agent, src_addr, src_agent, size, 0,
                                     NULL, rocm_ipc_signal->signal);

    if (status != HSA_STATUS_SUCCESS) {
        ucs_error("copy error");
        ucs_mpool_put(rocm_ipc_signal);
        return UCS_ERR_IO_ERROR;
    }

    rocm_ipc_signal->comp = comp;
    rocm_ipc_signal->mapped_addr = remote_base_addr;
    ucs_queue_push(&iface->signal_queue, &rocm_ipc_signal->queue);

    ucs_trace("rocm async copy issued :%p remote:%p, local:%p  len:%ld",
              rocm_ipc_signal, (void *)remote_addr, local_addr, size);

    return UCS_INPROGRESS;
}

ucs_status_t uct_rocm_ipc_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    ucs_status_t ret;
    uct_rocm_ipc_key_t *key = (uct_rocm_ipc_key_t *)rkey;

    ret = UCS_PROFILE_CALL_ALWAYS(uct_rocm_ipc_ep_zcopy, tl_ep, remote_addr,
                                  iov, key, comp, 1);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_rocm_ipc_trace_data(remote_addr, rkey, "PUT_ZCOPY [length %zu]",
                            uct_iov_total_length(iov, iovcnt));

    return ret;
}

ucs_status_t uct_rocm_ipc_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    ucs_status_t ret;
    uct_rocm_ipc_key_t *key = (uct_rocm_ipc_key_t *)rkey;

    ret = UCS_PROFILE_CALL_ALWAYS(uct_rocm_ipc_ep_zcopy, tl_ep, remote_addr,
                                  iov, key, comp, 0);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_rocm_ipc_trace_data(remote_addr, rkey, "GET_ZCOPY [length %zu]",
                            uct_iov_total_length(iov, iovcnt));

    return ret;
}
