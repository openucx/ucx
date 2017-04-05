/*
 * Copyright 2016 - 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
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

   ucs_trace("uct_rocm_cma_ep init class. Interface address: 0x%x", self->remote_pid);

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

ucs_status_t uct_rocm_cma_ep_common_zcopy(uct_ep_h tl_ep,
                                            const uct_iov_t *iov,
                                            size_t iovcnt,
                                            uint64_t remote_addr,
                                            uct_rocm_cma_key_t *key,
                                     HSAKMT_STATUS (*fn_p)(HSAuint32,
	                                                 HsaMemoryRange *,
	                                                 HSAuint64,
	                                                 HsaMemoryRange *,
	                                                 HSAuint64,
                                                     HSAuint64 *),
                                     char *fn_name)
{
    /* The logic was copied more/less verbatim from corresponding CMA zcopy
       function.
     */
    HSAuint64 delivered = 0;
    HSAuint64 SizeCopied;
    size_t iov_it;
    size_t iov_it_length;
    size_t iov_slice_length;
    size_t iov_slice_delivered;
    size_t local_iov_it;
    size_t length = 0;
    HsaMemoryRange local_iov[UCT_SM_MAX_IOV];
    HsaMemoryRange remote_iov;

    void   *local_ptr[UCT_SM_MAX_IOV];
    int     local_ptr_locked[UCT_SM_MAX_IOV];


    uct_rocm_cma_ep_t    *ep      = ucs_derived_of(tl_ep, uct_rocm_cma_ep_t);
    uct_rocm_cma_iface_t *iface   = ucs_derived_of(tl_ep->iface, uct_rocm_cma_iface_t);
    uct_rocm_cma_md_t    *rocm_md = (uct_rocm_cma_md_t *)iface->super.md;


    ucs_trace("uct_rocm_cma_ep_common_zcopy (%s): remote_addr: %p (gpu %p)",
                fn_name, (void *)remote_addr, (void*)key->address);

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
            ucs_status_t ucs_status = uct_rocm_cma_ptr_to_gpu_ptr(local_ptr[local_iov_it],
                                            &local_iov[local_iov_it].MemoryAddress,
                                            local_iov[local_iov_it].SizeInBytes,
                                            rocm_md->any_memory,
                                            &local_ptr_locked[local_iov_it]);

            if (ucs_status != UCS_OK) {
                uct_rocm_cma_unlock_ptrs(local_ptr, local_ptr_locked, local_iov_it);
                return ucs_status;
            }

            ucs_trace("uct_rocm_cma_ep_common_zcopy: [%d] Local address %p (GPU ptr %p), Local Size 0x%x",
                                (int) local_iov_it,
                                local_ptr[local_iov_it],
                                local_iov[local_iov_it].MemoryAddress,
                                (int) local_iov[local_iov_it].SizeInBytes);

            ++local_iov_it;
        }

        if (!delivered) {
            length = iov_it_length; /* Keep total length of the iov buffers */
        }

        if(!length) {
            return UCS_OK;          /* Nothing to deliver */
        }

        remote_iov.MemoryAddress = (void *)(key->address + delivered);
        remote_iov.SizeInBytes   = length - delivered;

        HSAKMT_STATUS hsa_status = fn_p(ep->remote_pid,
                                        local_iov, local_iov_it,
                                        &remote_iov, 1,
                                        &SizeCopied);

        uct_rocm_cma_unlock_ptrs(local_ptr, local_ptr_locked, local_iov_it);

        if (hsa_status  != HSAKMT_STATUS_SUCCESS) {
            ucs_error("%s  copied  %zu instead of %zu, Status  %d",
                         fn_name, (ssize_t) SizeCopied, (ssize_t) length,
                         hsa_status);
            return UCS_ERR_IO_ERROR;
        }

        delivered += SizeCopied;
    } while (delivered < length);

    return UCS_OK;
}

ucs_status_t uct_rocm_cma_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                  uint64_t remote_addr, uct_rkey_t rkey,
                                  uct_completion_t *comp)
{
    uct_rocm_cma_key_t *key = (uct_rocm_cma_key_t *)rkey;

    ucs_trace("uct_rocm_cma_ep_put_zcopy()");

    UCT_CHECK_IOV_SIZE(iovcnt, uct_sm_get_max_iov(), "uct_rocm_cma_ep_put_zcopy");

    ucs_status_t ret = uct_rocm_cma_ep_common_zcopy(tl_ep, iov,  iovcnt,
                                           remote_addr,
                                           key,
                                           hsaKmtProcessVMWrite,
                                           "hsaKmtProcessVMWrite");

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_rocm_cma_trace_data(remote_addr, rkey, "PUT_ZCOPY [length %zu]",
                       uct_iov_total_length(iov, iovcnt));

    ucs_trace("uct_rocm_cma_ep_put_zcopy(). Status: 0x%x", ret);
    return ret;
}

ucs_status_t uct_rocm_cma_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                  uint64_t remote_addr, uct_rkey_t rkey,
                                  uct_completion_t *comp)
{
    uct_rocm_cma_key_t *key = (uct_rocm_cma_key_t *)rkey;

    ucs_trace("uct_rocm_cma_ep_get_zcopy()");

    UCT_CHECK_IOV_SIZE(iovcnt, uct_sm_get_max_iov(), "uct_rocm_cma_ep_get_zcopy");


    ucs_status_t ret = uct_rocm_cma_ep_common_zcopy(tl_ep, iov,  iovcnt,
                                                    remote_addr,
                                                    key,
                                                    hsaKmtProcessVMRead,
                                                    "hsaKmtProcessVMRead");

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_rocm_cma_trace_data(remote_addr, rkey, "GET_ZCOPY [length %zu]",
                       uct_iov_total_length(iov, iovcnt));

    ucs_trace("uct_rocm_cma_ep_get_zcopy(). Status: 0x%x", ret);

    return ret;
}
