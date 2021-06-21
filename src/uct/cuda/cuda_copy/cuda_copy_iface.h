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


#define UCT_CUDA_COPY_IFACE_DEFAULT_BANDWIDTH (10000.0 * UCS_MBYTE)


#define UCT_CUDA_COPY_IFACE_OVERHEAD          (0)


#define uct_cuda_copy_for_each_q_desc(iface, q_var, code) { \
    int __i,__j; \
    for (__i = 0; __i < UCS_MEMORY_TYPE_LAST; ++__i) { \
        for (__j = 0; __j < UCS_MEMORY_TYPE_LAST; ++__j) { \
            (q_var) = &iface->queue_desc[__i][__j]; \
            code; \
        } \
    } \
}


#define uct_cuda_copy_for_each_stream(iface, stream_var, code) { \
    int __i,__j; \
    for (__i = 0; __i < UCS_MEMORY_TYPE_LAST; ++__i) { \
        for (__j = 0; __j < UCS_MEMORY_TYPE_LAST; ++__j) { \
            (stream_var) = &iface->queue_desc[__i][__j].stream; \
            code; \
        } \
    } \
}

#define uct_cuda_copy_for_each_event_q(iface, q_var, code) { \
    int __i,__j; \
    for (__i = 0; __i < UCS_MEMORY_TYPE_LAST; ++__i) { \
        for (__j = 0; __j < UCS_MEMORY_TYPE_LAST; ++__j) { \
            (q_var) = &iface->queue_desc[__i][__j].outstanding_event_q; \
            code; \
        } \
    } \
}

#define uct_cuda_copy_for_each_stream_event_q(iface, stream_var, q_var, code) { \
    int __i,__j; \
    for (__i = 0; __i < UCS_MEMORY_TYPE_LAST; ++__i) { \
        for (__j = 0; __j < UCS_MEMORY_TYPE_LAST; ++__j) { \
            (stream_var) = &iface->queue_desc[__i][__j].stream; \
            (q_var)      = &iface->queue_desc[__i][__j].outstanding_event_q; \
            code; \
        } \
    } \
}


typedef uint64_t uct_cuda_copy_iface_addr_t;


typedef struct uct_cuda_copy_queue_desc {
    cudaStream_t                stream;
    ucs_queue_head_t            outstanding_event_q;
    ucs_queue_head_t            *active_queue;
    ucs_queue_elem_t            queue;
} uct_cuda_copy_queue_desc_t;


typedef struct uct_cuda_copy_iface {
    uct_base_iface_t            super;
    uct_cuda_copy_iface_addr_t  id;
    ucs_mpool_t                 cuda_event_desc;
    ucs_queue_head_t            active_queue;
    cudaStream_t                short_stream;
    uct_cuda_copy_queue_desc_t  queue_desc[UCS_MEMORY_TYPE_LAST][UCS_MEMORY_TYPE_LAST];
    struct {
        unsigned                max_poll;
        unsigned                max_cuda_events;
        size_t                  detect_thresh;
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
    size_t                  detect_thresh;
} uct_cuda_copy_iface_config_t;


typedef struct uct_cuda_copy_event_desc {
    cudaEvent_t event;
    uct_completion_t *comp;
    ucs_queue_elem_t  queue;
} uct_cuda_copy_event_desc_t;
#endif
