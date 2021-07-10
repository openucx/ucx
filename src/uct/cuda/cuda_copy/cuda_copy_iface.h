/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_COPY_IFACE_H
#define UCT_CUDA_COPY_IFACE_H

#include <uct/base/uct_iface.h>
#include <uct/cuda/base/cuda_iface.h>
#include <ucs/memory/memtype_cache.h>
#include <pthread.h>


#define UCT_CUDA_COPY_IFACE_DEFAULT_BANDWIDTH (10000.0 * UCS_MBYTE)


#define UCT_CUDA_COPY_IFACE_OVERHEAD          (0)


typedef uint64_t uct_cuda_copy_iface_addr_t;


typedef enum uct_cuda_copy_stream {
    UCT_CUDA_COPY_STREAM_H2D,
    UCT_CUDA_COPY_STREAM_D2H,
    UCT_CUDA_COPY_STREAM_LAST
} uct_cuda_copy_stream_t;


typedef struct uct_cuda_copy_iface {
    uct_base_iface_t            super;
    uct_cuda_copy_iface_addr_t  id;
    ucs_memtype_cache_t         *memtype_cache;
    ucs_mpool_t                 cuda_event_desc;
    ucs_queue_head_t            outstanding_event_q[UCT_CUDA_COPY_STREAM_LAST];
    cudaStream_t                stream[UCT_CUDA_COPY_STREAM_LAST];
    struct {
        unsigned                max_poll;
        unsigned                max_cuda_events;
        int                     enable_memtype_cache;
    } config;
    struct {
        void                    *event_arg;
        uct_async_event_cb_t    event_cb;
    } async;
} uct_cuda_copy_iface_t;


typedef struct uct_cuda_copy_iface_config {
    uct_iface_config_t      super;
    unsigned                max_poll;
    unsigned                max_cuda_events;
    int                     enable_memtype_cache;
} uct_cuda_copy_iface_config_t;


typedef struct uct_cuda_copy_event_desc {
    cudaEvent_t event;
    uct_completion_t *comp;
    ucs_queue_elem_t  queue;
} uct_cuda_copy_event_desc_t;


void uct_cuda_copy_memory_detect(uct_cuda_copy_iface_t *iface,
                                 const void *address, size_t length,
                                 ucs_memory_info_t *mem_info);
#endif
