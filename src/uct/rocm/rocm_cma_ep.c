/*
 * Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */


#include "rocm_cma_ep.h"
#include "rocm_cma_md.h"
#include "rocm_common.h"

/* Include HSA Thunk header file for CMA API*/
#include <hsakmt.h>

#include <uct/base/uct_log.h>
#include <ucs/debug/memtrack.h>
#include <uct/sm/base/sm_iface.h>


static UCS_CLASS_INIT_FUNC(uct_rocm_cma_ep_t, uct_iface_t *tl_iface,
                           const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)
{
    uct_rocm_cma_iface_t *iface = ucs_derived_of(tl_iface, uct_rocm_cma_iface_t);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    self->remote_pid = *(const pid_t*)iface_addr;

    ucs_trace("uct_rocm_cma_ep init class. Interface address (pid): 0x%x", self->remote_pid);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rocm_cma_ep_t)
{
    /* No op */
}

UCS_CLASS_DEFINE(uct_rocm_cma_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rocm_cma_ep_t, uct_ep_t, uct_iface_t*,
                          const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rocm_cma_ep_t, uct_ep_t);


#define uct_rocm_cma_trace_data(_remote_addr, _rkey, _fmt, ...) \
    ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_remote_addr), \
                   (_rkey))


/** Convert pointer to pointer which could be used for 
  * GPU access.
*/
static ucs_status_t uct_rocm_cma_ptr_to_gpu_ptr(void *ptr, void **gpu_address,
                                                size_t size, int any_memory,
                                                int *locked)
{
    /* Assume that we do not need to lock memory */
    *locked = 0;

    /* Try to get GPU address if any */
    if (!uct_rocm_is_ptr_gpu_accessible(ptr, gpu_address)) {
        /* We do not have GPU address. Check what to do. */
        if (!any_memory) {
            /* We do not want to deal with memory about which
             * ROCm stack is not aware */
            ucs_warn("Address %p is not GPU registered", ptr);
            return UCS_ERR_INVALID_ADDR;
        } else {
            /* Register / lock this memory for GPU access */
            hsa_status_t status =  uct_rocm_memory_lock(ptr, size, gpu_address);

            if (status != HSA_STATUS_SUCCESS) {
                ucs_error("Could not lock  %p. Status %d", ptr, status);
                return UCS_ERR_INVALID_ADDR;
            } else {
                ucs_trace("Lock address %p as GPU %p", ptr, *gpu_address);
                /* We locked this memory. Set the flag to be aware that
                 * we need to unlock it later */
                *locked = 1;
            }
        }
    }

    return UCS_OK;
}

/** Release GPU address if it was previously locked */
static void uct_rocm_cma_unlock_ptrs(void **local_ptr, int *locked,
                                     size_t local_iov_it)
{
    size_t index;

    for (index = 0; index < local_iov_it; index++) {
        /* Check if memory was locked by us */
        if (locked[index]) {
            hsa_status_t status = hsa_amd_memory_unlock(local_ptr[index]);

            if (status != HSA_STATUS_SUCCESS) {
                ucs_warn("Failed to unlock memory (%p): 0x%x\n",
                         local_ptr[index], status);
            }
        }
    }
}

/**
 * Typedef for thunk CMA API.
 */
typedef HSAKMT_STATUS (*uct_rocm_hsakmt_cma_t)(HSAuint32,
                                               HsaMemoryRange *, HSAuint64,
                                               HsaMemoryRange *, HSAuint64,
                                               HSAuint64 *);

ucs_status_t uct_rocm_cma_ep_common_zcopy(uct_ep_h tl_ep,
                                          const uct_iov_t *iov,
                                          size_t iovcnt,
                                          uint64_t remote_addr,
                                          uct_rocm_cma_key_t *key,
                                          uct_rocm_hsakmt_cma_t fn_p,
                                          char *fn_name)
{
    /* The logic was copied more/less verbatim from corresponding CMA zcopy
       function.
     */
    HSAuint64 delivered = 0;
    HSAuint64 size_copied;
    size_t iov_it;
    size_t iov_it_length;
    size_t iov_slice_length;
    size_t iov_slice_delivered;
    size_t local_iov_it;
    size_t length = 0;
    HsaMemoryRange local_iov[UCT_SM_MAX_IOV];
    HsaMemoryRange remote_iov;
    ucs_status_t ucs_status;
    HSAKMT_STATUS hsa_status;
    void   *local_ptr[UCT_SM_MAX_IOV];
    int     local_ptr_locked[UCT_SM_MAX_IOV];
    uint64_t remote_gpu_address;

    uct_rocm_cma_ep_t    *ep      = ucs_derived_of(tl_ep, uct_rocm_cma_ep_t);
    uct_rocm_cma_iface_t *iface   = ucs_derived_of(tl_ep->iface, uct_rocm_cma_iface_t);
    uct_rocm_cma_md_t    *rocm_md = (uct_rocm_cma_md_t *)iface->super.md;

    /*  Check if we do not have 0 size transfer. By design if length of transfer
        is 0 then all other parameters could be invalid (including pointers)
        and should not be touched. To deal with GPU memory we rely on information
        passed in the key. To simplify the logic and assuming that performance impact
        of several extra CPU operations will be immaterial for the whole operation
        we do validation in the beginning of function as independent and atomic
        operation */

    for (iov_it = 0; iov_it < ucs_min(UCT_SM_MAX_IOV, iovcnt); ++iov_it) {
        if (uct_iov_get_length(iov + iov_it)) {
            /* Found element with some length */
            break;
        }
    }

    if (iov_it == ucs_min(UCT_SM_MAX_IOV, iovcnt)) {
        /* We reach the end of array and have nothing to deliver */
        return UCS_OK;
    }


    ucs_trace("(%s): remote_addr: %p (gpu %p, md addr %p), key->length 0x%x",
              fn_name, (void *)remote_addr, (void*)key->gpu_address,
              (void*)key->md_address, (int) key->length);

    /* Remote_addr could be inside of rkey range. We need
       to calculate the correct gpu address based on the
       passed address in case if it is non-GPU ones.*/

    if ((remote_addr < key->md_address) ||
        (remote_addr > (key->md_address + key->length))) {
        ucs_error("remote_addr %p is out of range [%p, +0x%x]",
                  (void *)remote_addr, (void *)key->md_address, (int) key->length);
        return UCS_ERR_INVALID_PARAM;
    }

    remote_gpu_address = key->gpu_address + (remote_addr - key->md_address);

    do {
        iov_it_length = 0;
        local_iov_it = 0;
        for (iov_it = 0; iov_it < ucs_min(UCT_SM_MAX_IOV, iovcnt); ++iov_it) {
            iov_slice_delivered = 0;

            /* Get length of the particular iov element */
            iov_slice_length = uct_iov_get_length(iov + iov_it);

            /* Skip the iov element if no data */
            if (!iov_slice_length) {
                continue;
            }
            iov_it_length += iov_slice_length;

            if (iov_it_length <= delivered) {
                continue; /* Skip the iov element if transferred already */
            } else {
                /* Let's assume the iov element buffer can be delivered partially */
                if ((iov_it_length - delivered) < iov_slice_length) {
                    iov_slice_delivered = iov_slice_length - (iov_it_length - delivered);
                }
            }

            local_ptr[local_iov_it]             = (void *)((char *)iov[iov_it].buffer +
                                                           iov_slice_delivered);
            local_iov[local_iov_it].SizeInBytes = iov_slice_length - iov_slice_delivered;

            /* It is possible that we get host (CPU) address as local address.
             * We need to get GPU address to be used for CMA operation.
             * If this is memory was not yet registered with ROCm stack and
             * flag "any_memory" is set than lock this memory.
             */
            ucs_status = uct_rocm_cma_ptr_to_gpu_ptr(local_ptr[local_iov_it],
                                                     &local_iov[local_iov_it].MemoryAddress,
                                                     local_iov[local_iov_it].SizeInBytes,
                                                     rocm_md->any_memory,
                                                     &local_ptr_locked[local_iov_it]);

            if (ucs_status != UCS_OK) {
                uct_rocm_cma_unlock_ptrs(local_ptr, local_ptr_locked, local_iov_it);
                return ucs_status;
            }

            ucs_trace("[%d] Local address %p (GPU ptr %p), Local Size 0x%x",
                      (int) local_iov_it, local_ptr[local_iov_it],
                      local_iov[local_iov_it].MemoryAddress,
                      (int) local_iov[local_iov_it].SizeInBytes);

            ++local_iov_it;
        }

        if (!delivered) {
            length = iov_it_length; /* Keep total length of the iov buffers */
        }

        if (!length) {
            ucs_trace("Nothing to zcopy");
            return UCS_OK; /* Nothing to deliver */
        }

        remote_iov.MemoryAddress = (void *)(remote_gpu_address + delivered);
        remote_iov.SizeInBytes   = length - delivered;

        ucs_trace("remote_iov.MemoryAddress %p, remote_iov.SizeInBytes 0x%x",
                  remote_iov.MemoryAddress, (int) remote_iov.SizeInBytes);

        hsa_status = fn_p(ep->remote_pid, local_iov, local_iov_it,
                          &remote_iov, 1, &size_copied);

        uct_rocm_cma_unlock_ptrs(local_ptr, local_ptr_locked, local_iov_it);

        if (hsa_status  != HSAKMT_STATUS_SUCCESS) {
            ucs_error("%s  copied  %zu instead of %zu, Status  %d. errno %d",
                      fn_name, (ssize_t) size_copied, (ssize_t) length,
                      hsa_status, errno);
            return UCS_ERR_IO_ERROR;
        }

        delivered += size_copied;
    } while (delivered < length);

    return UCS_OK;
}

ucs_status_t uct_rocm_cma_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    ucs_status_t ret;
    uct_rocm_cma_key_t *key = (uct_rocm_cma_key_t *)rkey;

    ucs_trace_func("");

    UCT_CHECK_IOV_SIZE(iovcnt, uct_sm_get_max_iov(), "uct_rocm_cma_ep_put_zcopy");

    ret = uct_rocm_cma_ep_common_zcopy(tl_ep, iov,  iovcnt,
                                       remote_addr, key,
                                       hsaKmtProcessVMWrite,
                                       "hsaKmtProcessVMWrite");

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_rocm_cma_trace_data(remote_addr, rkey, "PUT_ZCOPY [length %zu]",
                            uct_iov_total_length(iov, iovcnt));

    ucs_trace("put zcopy status: 0x%x", ret);
    return ret;
}

ucs_status_t uct_rocm_cma_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    ucs_status_t ret;
    uct_rocm_cma_key_t *key = (uct_rocm_cma_key_t *)rkey;

    ucs_trace_func("");

    UCT_CHECK_IOV_SIZE(iovcnt, uct_sm_get_max_iov(), "uct_rocm_cma_ep_get_zcopy");

    ret = uct_rocm_cma_ep_common_zcopy(tl_ep, iov, iovcnt,
                                       remote_addr, key,
                                       hsaKmtProcessVMRead,
                                       "hsaKmtProcessVMRead");

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_rocm_cma_trace_data(remote_addr, rkey, "GET_ZCOPY [length %zu]",
                            uct_iov_total_length(iov, iovcnt));

    ucs_trace("get zcopy status: 0x%x", ret);
    return ret;
}
