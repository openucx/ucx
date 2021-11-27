/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_IFACE_H
#define UCT_CUDA_IPC_IFACE_H

#include <uct/base/uct_iface.h>
#include <ucs/arch/cpu.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "cuda_ipc_md.h"
#include "cuda_ipc_ep.h"
#include "cuda_ipc_cache.h"


#define UCT_CUDA_IPC_MAX_PEERS  16

typedef struct uct_cuda_ipc_iface {
    uct_base_iface_t super;
    ucs_mpool_t      event_desc;              /* cuda event desc */
    ucs_queue_head_t outstanding_d2d_event_q; /* stream for outstanding d2d */
    int              eventfd;              /* get event notifications */
    int              streams_initialized;     /* indicates if stream created */
    CUcontext        cuda_context;
    CUstream         stream_d2d[UCT_CUDA_IPC_MAX_PEERS];
                                              /* per-peer stream */
    unsigned long    stream_refcount[UCT_CUDA_IPC_MAX_PEERS];
                                              /* per stream outstanding ops */
    struct {
        unsigned                max_poll;            /* query attempts w.o success */
        unsigned                max_streams;         /* # concurrent streams for || progress*/
        unsigned                max_cuda_ipc_events; /* max mpool entries */
        int                     enable_cache;        /* enable/disable ipc handle cache */
        ucs_on_off_auto_value_t enable_get_zcopy;    /* enable get_zcopy except for specific platorms */
    } config;
} uct_cuda_ipc_iface_t;


typedef struct uct_cuda_ipc_iface_config {
    uct_iface_config_t      super;
    unsigned                max_poll;
    unsigned                max_streams;
    int                     enable_cache;
    ucs_on_off_auto_value_t enable_get_zcopy;
    unsigned                max_cuda_ipc_events;
} uct_cuda_ipc_iface_config_t;


typedef struct uct_cuda_ipc_event_desc {
    CUevent           event;
    void              *mapped_addr;
    unsigned          stream_id;
    uct_completion_t  *comp;
    ucs_queue_elem_t  queue;
    uct_cuda_ipc_ep_t *ep;
    uintptr_t         d_bptr;
    pid_t             pid;
} uct_cuda_ipc_event_desc_t;


ucs_status_t uct_cuda_ipc_iface_init_streams(uct_cuda_ipc_iface_t *iface);
#endif
