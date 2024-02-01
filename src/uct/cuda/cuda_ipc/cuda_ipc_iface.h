/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018. ALL RIGHTS RESERVED.
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


#define UCT_CUDA_IPC_MAX_PEERS  16


typedef struct uct_cuda_ipc_iface {
    uct_cuda_iface_t super;
    ucs_mpool_t      event_desc;              /* cuda event desc */
    int              eventfd;              /* get event notifications */
    int              streams_initialized;     /* indicates if stream created */
    CUcontext        cuda_context;
    ucs_queue_head_t      active_queue;
    uct_cuda_queue_desc_t queue_desc[UCT_CUDA_IPC_MAX_PEERS];
    struct {
        unsigned                max_poll;            /* query attempts w.o success */
        unsigned                max_streams;         /* # concurrent streams for || progress*/
        unsigned                max_cuda_ipc_events; /* max mpool entries */
        int                     enable_cache;        /* enable/disable ipc handle cache */
        ucs_on_off_auto_value_t enable_get_zcopy;    /* enable get_zcopy except for specific platorms */
        double                  bandwidth;
    } config;
} uct_cuda_ipc_iface_t;


typedef struct uct_cuda_ipc_iface_config {
    uct_iface_config_t      super;
    unsigned                max_poll;
    unsigned                max_streams;
    int                     enable_cache;
    ucs_on_off_auto_value_t enable_get_zcopy;
    unsigned                max_cuda_ipc_events;
    double                  bandwidth;
} uct_cuda_ipc_iface_config_t;


typedef struct uct_cuda_ipc_event_desc {
    CUevent           event;
    void              *mapped_addr;
    uct_completion_t  *comp;
    ucs_queue_elem_t  queue;
    uct_cuda_ipc_ep_t *ep;
    uintptr_t         d_bptr;
    pid_t             pid;
} uct_cuda_ipc_event_desc_t;


ucs_status_t uct_cuda_ipc_iface_init_streams(uct_cuda_ipc_iface_t *iface);
#endif
