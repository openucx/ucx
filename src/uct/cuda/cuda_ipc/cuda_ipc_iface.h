/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018-2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_IFACE_H
#define UCT_CUDA_IPC_IFACE_H

#include <uct/base/uct_iface.h>
#include <uct/cuda/base/cuda_iface.h>
#include <ucs/arch/cpu.h>
#include <cuda.h>

#include "cuda_ipc_md.h"
#include "cuda_ipc_ep.h"
#include "cuda_ipc_cache.h"


#define UCT_CUDA_IPC_MAX_PEERS 128


typedef struct {
    unsigned                max_poll;            /* query attempts w.o success */
    unsigned                max_streams;         /* # concurrent streams for || progress*/
    unsigned                max_cuda_ipc_events; /* max mpool entries */
    int                     enable_cache;        /* enable/disable ipc handle cache */
    ucs_on_off_auto_value_t enable_get_zcopy;    /* enable get_zcopy except for specific platforms */
    double                  bandwidth;           /* estimated bandwidth */
    double                  latency;             /* estimated latency */
    double                  overhead;            /* estimated CPU overhead */
} uct_cuda_ipc_iface_config_params_t;


typedef struct {
    uct_cuda_iface_t                   super;
    uct_cuda_ipc_iface_config_params_t config;
} uct_cuda_ipc_iface_t;


typedef struct {
    uct_iface_config_t                 super;
    uct_cuda_ipc_iface_config_params_t params;
} uct_cuda_ipc_iface_config_t;


#if CUDA_VERSION >= 13000
typedef struct {
    size_t       count;
    pid_t        *pids;
    ucs_sys_ns_t *pid_nss;
    uintptr_t    *d_bptrs;
    void         **mapped_addrs;
} uct_cuda_ipc_sgl_mapping_t;


static UCS_F_ALWAYS_INLINE void
uct_cuda_ipc_sgl_unmap(uct_cuda_ipc_sgl_mapping_t *mapping,
                       size_t count, CUdevice cuda_device,
                       int enable_cache)
{
    size_t i;
    ucs_status_t status;

    for (i = 0; i < count; i++) {
        status = uct_cuda_ipc_unmap_memhandle(mapping->pids[i],
                                              mapping->pid_nss[i],
                                              mapping->d_bptrs[i],
                                              mapping->mapped_addrs[i],
                                              cuda_device, enable_cache);
        if (status != UCS_OK) {
            ucs_fatal("failed to unmap addr:%p", mapping->mapped_addrs[i]);
        }
    }
}
#endif


typedef struct {
    uct_cuda_event_desc_t super;
    void                  *mapped_addr;
    uct_cuda_ipc_ep_t     *ep;
    uintptr_t             d_bptr;
    pid_t                 pid;
    ucs_sys_ns_t          pid_ns;
    CUdevice              cuda_device;
#if CUDA_VERSION >= 13000
    uct_cuda_ipc_sgl_mapping_t *sgl_mapping;
#endif
} uct_cuda_ipc_event_desc_t;


typedef struct {
    uct_cuda_ctx_rsc_t    super;
    uct_cuda_queue_desc_t queue_desc[UCT_CUDA_IPC_MAX_PEERS];
} uct_cuda_ipc_ctx_rsc_t;

#endif
