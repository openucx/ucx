/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_COPY_IFACE_H
#define UCT_CUDA_COPY_IFACE_H

#include <uct/base/uct_iface.h>
#include <uct/cuda/base/cuda_iface.h>
#include <ucs/memory/memory_type.h>
#include <pthread.h>


typedef enum uct_cuda_copy_loc {
    UCT_CUDA_COPY_LOC_HOST,
    UCT_CUDA_COPY_LOC_PINNED,
    UCT_CUDA_COPY_LOC_MANAGED,
    UCT_CUDA_COPY_LOC_LAST
} uct_cuda_copy_loc_t;


#define uct_cuda_copy_get_mem_type_index(mem_type, index) { \
    switch (mem_type) { \
        case UCS_MEMORY_TYPE_HOST: \
            index = UCT_CUDA_COPY_LOC_HOST; \
            break; \
        case UCS_MEMORY_TYPE_CUDA: \
            index = UCT_CUDA_COPY_LOC_PINNED; \
            break; \
        case UCS_MEMORY_TYPE_CUDA_MANAGED: \
            index = UCT_CUDA_COPY_LOC_MANAGED; \
            break; \
        default: \
            index = UCT_CUDA_COPY_LOC_LAST; \
            break; \
    } \
}


#define uct_cuda_copy_for_each_stream(iface, stream_var, code) { \
    int __i,__j; \
    for (__i = 0; __i < UCT_CUDA_COPY_LOC_LAST; ++__i) { \
        for (__j = 0; __j < UCT_CUDA_COPY_LOC_LAST; ++__j) { \
            (stream_var) = &iface->stream[__i][__j]; \
            code; \
        } \
    } \
}

#define uct_cuda_copy_for_each_event_q(iface, q_var, code) { \
    int __i,__j; \
    for (__i = 0; __i < UCT_CUDA_COPY_LOC_LAST; ++__i) { \
        for (__j = 0; __j < UCT_CUDA_COPY_LOC_LAST; ++__j) { \
            (q_var) = &iface->outstanding_event_q[__i][__j]; \
            code; \
        } \
    } \
}

#define uct_cuda_copy_for_each_stream_event_q(iface, stream_var, q_var, code) { \
    int __i,__j; \
    for (__i = 0; __i < UCT_CUDA_COPY_LOC_LAST; ++__i) { \
        for (__j = 0; __j < UCT_CUDA_COPY_LOC_LAST; ++__j) { \
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
    ucs_mpool_t                 cuda_event_desc;
    ucs_queue_head_t            outstanding_event_q[UCT_CUDA_COPY_LOC_LAST][UCT_CUDA_COPY_LOC_LAST];
    cudaStream_t                stream[UCT_CUDA_COPY_LOC_LAST][UCT_CUDA_COPY_LOC_LAST];
    struct {
        unsigned                max_poll;
        unsigned                max_cuda_events;
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
} uct_cuda_copy_iface_config_t;


typedef struct uct_cuda_copy_event_desc {
    cudaEvent_t event;
    uct_completion_t *comp;
    ucs_queue_elem_t  queue;
} uct_cuda_copy_event_desc_t;

#endif
