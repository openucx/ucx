/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_COPY_IFACE_H
#define UCT_CUDA_COPY_IFACE_H

#include <uct/base/uct_iface.h>
#include <uct/cuda/base/cuda_iface.h>
#include <pthread.h>


typedef uint64_t uct_cuda_copy_iface_addr_t;


typedef enum uct_cuda_copy_stream {
    UCT_CUDA_COPY_H2D_STREAM  = 0,
    UCT_CUDA_COPY_D2H_STREAM  = 1,
    UCT_CUDA_COPY_LAST_STREAM = 2
} uct_cuda_copy_stream_t;


typedef struct uct_cuda_copy_iface {
    uct_base_iface_t            super;
    uct_cuda_copy_iface_addr_t  id;
    ucs_mpool_t                 cuda_event_desc;
    ucs_queue_head_t            outstanding_event_q[UCT_CUDA_COPY_LAST_STREAM];
    cudaStream_t                stream[UCT_CUDA_COPY_LAST_STREAM];
    unsigned long               stream_refcount[UCT_CUDA_COPY_LAST_STREAM];
    struct {
        unsigned                max_poll;
        unsigned                max_cuda_events;
    } config;
    struct {
        uct_async_event_cb_t    event_cb;
        void                    *event_arg;
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
    unsigned          stream_id;
} uct_cuda_copy_event_desc_t;

#endif
