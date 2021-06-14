/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_COPY_IFACE_H
#define UCT_CUDA_COPY_IFACE_H

#include <uct/base/uct_iface.h>
#include <uct/cuda/base/cuda_iface.h>
#include <ucs/memory/memtype_cache.h>
#include <ucs/memory/memory_type.h>
#include <pthread.h>


#define UCT_CUDA_COPY_IFACE_DEFAULT_BANDWIDTH (10000.0 * UCS_MBYTE)


#define UCT_CUDA_COPY_IFACE_OVERHEAD          (0)


#define uct_cuda_copy_for_each_stream(iface, stream_var, code) { \
    int __i,__j; \
    for (__i = 0; __i < UCS_MEMORY_TYPE_LAST; ++__i) { \
        for (__j = 0; __j < UCS_MEMORY_TYPE_LAST; ++__j) { \
            (stream_var) = &iface->stream[__i][__j]; \
            code; \
        } \
    } \
}

#define uct_cuda_copy_for_each_event_q(iface, q_var, code) { \
    int __i,__j; \
    for (__i = 0; __i < UCS_MEMORY_TYPE_LAST; ++__i) { \
        for (__j = 0; __j < UCS_MEMORY_TYPE_LAST; ++__j) { \
            (q_var) = &iface->outstanding_event_q[__i][__j]; \
            code; \
        } \
    } \
}

#define uct_cuda_copy_for_each_stream_event_q(iface, stream_var, q_var, code) { \
    int __i,__j; \
    for (__i = 0; __i < UCS_MEMORY_TYPE_LAST; ++__i) { \
        for (__j = 0; __j < UCS_MEMORY_TYPE_LAST; ++__j) { \
            (stream_var) = &iface->stream[__i][__j]; \
            (q_var)      = &iface->outstanding_event_q[__i][__j]; \
            code; \
        } \
    } \
}


typedef uint64_t uct_cuda_copy_iface_addr_t;


typedef struct uct_cuda_copy_iface {
    uct_base_iface_t            super;
    uct_cuda_copy_iface_addr_t  id;
    ucs_memtype_cache_t         *memtype_cache;
    ucs_mpool_t                 cuda_event_desc;
    ucs_queue_head_t            outstanding_event_q[UCS_MEMORY_TYPE_LAST][UCS_MEMORY_TYPE_LAST];
    cudaStream_t                stream[UCS_MEMORY_TYPE_LAST][UCS_MEMORY_TYPE_LAST];
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
