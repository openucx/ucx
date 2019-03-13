/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "rocm_ipc_ep.h"
#include "rocm_ipc_iface.h"
#include "rocm_ipc_md.h"

#include <uct/rocm/base/rocm_base.h>

#include <hsakmt.h>

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
    size_t size = uct_iov_get_length(iov);
    ucs_status_t ret = UCS_OK;
    void *lock_addr, *base_addr, *local_addr;

    /* no data to deliver */
    if (!size)
        return UCS_OK;

    if ((remote_addr < key->address) ||
        (remote_addr + size > key->address + key->length)) {
        ucs_error("remote addr %lx/%lx out of range %lx/%lx",
                  remote_addr, size, key->address, key->length);
        return UCS_ERR_INVALID_PARAM;
    }

    status = uct_rocm_base_lock_ptr(iov->buffer, size, &lock_addr,
                                    &base_addr, NULL, &local_agent);
    if (status != HSA_STATUS_SUCCESS)
        return status;

    local_addr = lock_addr ? lock_addr : iov->buffer;

    if (key->ipc_valid) {
        uct_rocm_ipc_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rocm_ipc_iface_t);
        void *remote_base_addr, *remote_copy_addr;
        void *dst_addr, *src_addr;
        hsa_agent_t dst_agent, src_agent;
        uct_rocm_ipc_signal_desc_t *rocm_ipc_signal;

        ret = uct_rocm_ipc_cache_map_memhandle((void *)ep->remote_memh_cache, key,
                                               &remote_base_addr);
        if (ret != UCS_OK) {
            ucs_error("fail to attach ipc mem %p %d\n", (void *)key->address, ret);
            goto out_unlock;
        }

        if (!lock_addr) {
            hsa_agent_t *gpu_agents;
            int num_gpu = uct_rocm_base_get_gpu_agents(&gpu_agents);
            status = hsa_amd_agents_allow_access(num_gpu, gpu_agents, NULL, base_addr);
            if (status != HSA_STATUS_SUCCESS) {
                ucs_error("fail to map local mem %p %p %d\n",
                          local_addr, base_addr, status);
                ret = UCS_ERR_INVALID_ADDR;
                goto out_unlock;
            }
        }

        remote_copy_addr = remote_base_addr + (remote_addr - key->address);
        remote_agent = uct_rocm_base_get_dev_agent(key->dev_num);

        if (is_put) {
            dst_addr = remote_copy_addr;
            dst_agent = remote_agent;

            src_addr = local_addr;
            src_agent = local_agent;
        }
        else {
            dst_addr = local_addr;
            dst_agent = local_agent;

            src_addr = remote_copy_addr;
            src_agent = remote_agent;
        }

        rocm_ipc_signal = ucs_mpool_get(&iface->signal_pool);
        hsa_signal_store_screlease(rocm_ipc_signal->signal, 1);

        status = hsa_amd_memory_async_copy(dst_addr, dst_agent,
                                           src_addr, src_agent,
                                           size, 0, NULL,
                                           rocm_ipc_signal->signal);

        if (status == HSA_STATUS_SUCCESS) {
            ret = UCS_INPROGRESS;
        }
        else {
            ucs_error("copy error");
            ret = UCS_ERR_IO_ERROR;
            ucs_mpool_put(rocm_ipc_signal);
            goto out_unlock;
        }

        rocm_ipc_signal->comp = comp;
        rocm_ipc_signal->mapped_addr = remote_base_addr;
        ucs_queue_push(&iface->signal_queue, &rocm_ipc_signal->queue);

        ucs_trace("rocm async copy issued :%p remote:%p, local:%p  len:%ld",
                  rocm_ipc_signal, (void *)remote_addr, local_addr, size);
    }
    else {
        /* fallback to cma when remote buffer has no ipc */
        HSAKMT_STATUS hsa_status;
        void *remote_copy_addr = key->lock_address ?
            (void *)key->lock_address + (remote_addr - key->address) :
            (void *)remote_addr;
        HsaMemoryRange local_mem = {
            .MemoryAddress = local_addr,
            .SizeInBytes = size,
        };
        HsaMemoryRange remote_mem = {
            .MemoryAddress = remote_copy_addr,
            .SizeInBytes = size,
        };
        HSAuint64 copied = 0;

        if (is_put)
            hsa_status = hsaKmtProcessVMWrite(ep->remote_pid, &local_mem, 1,
                                              &remote_mem, 1, &copied);
        else
            hsa_status = hsaKmtProcessVMRead(ep->remote_pid,  &local_mem, 1,
                                             &remote_mem, 1, &copied);

        if (hsa_status != HSAKMT_STATUS_SUCCESS) {
            ucs_error("cma copy fail %d %d", hsa_status, errno);
            ret = UCS_ERR_IO_ERROR;
        }
        else
            assert(copied == size);
    }

 out_unlock:
    if (lock_addr)
        hsa_amd_memory_unlock(lock_addr);

    return ret;
}

ucs_status_t uct_rocm_ipc_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    ucs_status_t ret;
    uct_rocm_ipc_key_t *key = (uct_rocm_ipc_key_t *)rkey;

    ret = uct_rocm_ipc_ep_zcopy(tl_ep, remote_addr, iov, key, comp, 1);

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

    ret = uct_rocm_ipc_ep_zcopy(tl_ep, remote_addr, iov, key, comp, 0);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_rocm_ipc_trace_data(remote_addr, rkey, "GET_ZCOPY [length %zu]",
                            uct_iov_total_length(iov, iovcnt));

    return ret;
}
